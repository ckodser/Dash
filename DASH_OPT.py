# DASH_OPT.py
# Main orchestrator for the DASH-OPT pipeline.

import os
import torch
from PIL import Image
from tqdm import tqdm
import json

import config
import pipeline_utils
from pipeline_utils import image_difference, unique_images
from data import COCO
from query_generator import DASH_LLM_QueryGenerator
from retrieval import Retriever
from filters import ObjectDetectorFilter, VLMFilter
from clustering import Clusterer
from optimizer import DASH_OPT_Optimizer


def setup_directories():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def run_pipeline(target_object: str, target_vlm: str):
    print(f"🚀 Starting DASH-OPT pipeline for object: '{target_object}' on VLM: '{target_vlm}'")

    # 1. SETUP & MODEL LOADING
    # ==========================
    print("\n[Phase 1/5] Setting up and loading models...")
    setup_directories()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models = pipeline_utils.load_models(target_vlm, device)
    llm_generator = models['llm_generator']
    vlm_filter_model = models['vlm']
    object_detector = models['object_detector']
    clip_model = models['clip']
    dreamsim_model = models['dreamsim']
    dreamsim_processor = models['dreamsim_processor']

    processors = pipeline_utils.load_processors()
    vlm_processor = processors['vlm']
    object_detector_processor = processors['object_detector']
    clip_processor = processors['clip']

    # 1.5. QUERY GENERATION
    # ===================
    print(f"\n[Phase 2/5] Generating text queries for '{target_object}'...")
    query_generator_dest = os.path.join(config.OUTPUT_DIR, f"{target_object}_queries.json")
    if os.path.exists(query_generator_dest):
        with open(query_generator_dest, 'r') as f:
            text_queries = json.load(f)
    else:
        query_gen = DASH_LLM_QueryGenerator(llm_generator['model'], llm_generator['tokenizer'], device)
        text_queries = query_gen.generate(target_object, num_queries=config.NUM_TEXT_QUERIES)
        print(f"Generated {len(text_queries)} unique text queries.")

        with open(query_generator_dest, 'w') as f:
            json.dump(text_queries, f, indent=2)


    # 2. IMAGE QUERY GENERATION (DASH-OPT)

    # Initialize the optimizer
    optimizer = DASH_OPT_Optimizer(
        vlm=models['vlm'],
        vlm_processor=processors['vlm'],
        object_detector=models['object_detector'],
        od_processor=processors['object_detector'],
        device=device
    )

    image_queries = []
    for prompt in tqdm(text_queries, desc="DASH-OPT Generation"):
        generated_image = optimizer.generate_image(prompt, target_object)
        if generated_image:
            image_queries.append(generated_image)

    print(f"Generated {len(image_queries)} image queries.")
    image_queries_with_path = pipeline_utils.save_images(
        image_queries,
        target_object,
        config.title + "DASH_OPT_Generated_Queries"
    )

    # 3. EXPLORATION PHASE
    print("\n[Phase 3/5] Running Exploration Phase with generated image queries...")
    dataset = COCO(config.DATASET_DIR)
    retriever = Retriever(
        dataset=dataset,
        clip_model=models['clip'],
        clip_processor=processors['clip'],
        dreamsim_model=models['dreamsim'],
        dreamsim_processor=models['dreamsim_processor'],
        device=device
    )
    retriever.build_index()

    vlm_filter = VLMFilter(models['vlm'], processors['vlm'], device)
    od_filter = ObjectDetectorFilter(models['object_detector'], processors['object_detector'], device)

    exploration_candidates = []
    for i,image_query in enumerate(tqdm(image_queries, desc="Exploration")):
        retrieved_images = retriever.retrieve_from_image(image_query, k=config.EXPLORATION_K)
        for j,image in enumerate(retrieved_images):
            if ((not od_filter.is_object_present(image, target_object) and vlm_filter.is_object_present(image, target_object))
                    or i==j==0):
                exploration_candidates.append(image)

    exploration_candidates = unique_images(exploration_candidates)
    print(f"Found {len(exploration_candidates)} successful candidates after Exploration.")
    exploration_candidates_embed = retriever.get_embeds(exploration_candidates)
    exploration_candidates_with_path = pipeline_utils.save_images(
        exploration_candidates,
        target_object,
        config.title + "Exploration_Phase_Results_DASH_OPT"
    )

    # 4. EXPLOITATION & 5. CLUSTERING (Identical to DASH_LLM)
    if not exploration_candidates:
        print("\nNo candidates found in exploration phase. Stopping pipeline.")
        return

    print("\n[Phase 4/5] Running Exploitation Phase...")
    exploitation_candidates_raw = []
    for i, image in enumerate(tqdm(exploration_candidates, desc="Exploitation")):
        retrieved_images = retriever.retrieve_from_image(image, k=config.EXPLOITATION_K)
        for j, retrieved_image in enumerate(retrieved_images):
            if ( not od_filter.is_object_present(retrieved_image, target_object) and vlm_filter.is_object_present(
                    retrieved_image, target_object)) or i==j==0:
                exploitation_candidates_raw.append(retrieved_image)

    exploitation_candidates_raw = image_difference(unique_images(exploitation_candidates_raw), exploration_candidates)
    print(f"Retrieved {len(exploitation_candidates_raw)} raw candidates. De-duplicating...")
    exploitation_candidates, exploitation_candidates_embed = retriever.deduplicate(exploitation_candidates_raw)
    print(f"Found {len(exploitation_candidates)} new, unique candidates after Exploitation.")
    exploitation_candidates_with_path = pipeline_utils.save_images(
        exploitation_candidates,
        target_object,
        config.title + "Exploitation_Phase_Results_DASH_OPT"
    )

    print("\n[Phase 5/5] Clustering final candidates...")
    all_successful_images_with_path = exploration_candidates_with_path + exploitation_candidates_with_path
    all_successful_images_embed = torch.cat([exploration_candidates_embed, exploitation_candidates_embed], dim=0)
    if len(all_successful_images_with_path) < 2:
        print("Not enough images to perform clustering.")
        return

    clusterer = Clusterer(
        dreamsim_model=models['dreamsim'],
        dreamsim_processor=models['dreamsim_processor'],
        device=device
    )
    clusters = clusterer.perform_clustering(all_successful_images_with_path, all_successful_images_embed)
    output_clusters_path = os.path.join(config.OUTPUT_DIR, f"{target_object}_clusters_OPT.json")
    with open(output_clusters_path, 'w') as f:
        json.dump(clusters, f, indent=2)

    print(f"Clustering complete. Found {len(clusters)} clusters.")
    print(f"Results saved to {output_clusters_path}")
    print("\n✅ DASH-OPT pipeline finished successfully!")


if __name__ == "__main__":
    TARGET_OBJECT = "leopard"
    TARGET_VLM_MODEL = config.VLM_MODEL_NAME
    run_pipeline(TARGET_OBJECT, TARGET_VLM_MODEL)