# This is the main orchestrator script for the DASH-LLM pipeline.
# It brings together all the components to find systematic hallucinations.

import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import json

# Import all our custom modules
import config
import utils
from data import COCO
from query_generator import DASH_LLM_QueryGenerator
from retrieval import Retriever
from filters import ObjectDetectorFilter, VLMFilter
from clustering import Clusterer  # MODIFIED: Uses DreamSim


def setup_directories():
    """Create necessary directories for outputs."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def run_pipeline(target_object: str, target_vlm: str):
    """
    Executes the full DASH-LLM pipeline for a given object and VLM.

    Args:
        target_object (str): The object to search for hallucinations of (e.g., "dining table").
        target_vlm (str): The Hugging Face model name of the VLM to test.
    """
    print(f"🚀 Starting DASH-LLM pipeline for object: '{target_object}' on VLM: '{target_vlm}'")

    # 1. SETUP & MODEL LOADING
    # ==========================
    print("\n[Phase 1/5] Setting up and loading models...")
    setup_directories()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    models = utils.load_models(target_vlm, device)
    llm_generator = models['llm_generator']
    vlm_filter_model = models['vlm']
    object_detector = models['object_detector']
    clip_model = models['clip']
    dreamsim_model = models['dreamsim']

    processors = utils.load_processors()
    vlm_processor = processors['vlm']
    object_detector_processor = processors['object_detector']
    clip_processor = processors['clip']

    # 2. QUERY GENERATION
    # ===================
    print(f"\n[Phase 2/5] Generating text queries for '{target_object}'...")
    query_gen = DASH_LLM_QueryGenerator(llm_generator['model'], llm_generator['tokenizer'], device)
    text_queries = query_gen.generate(target_object, num_queries=config.NUM_TEXT_QUERIES)
    print(f"Generated {len(text_queries)} unique text queries.")

    with open(os.path.join(config.OUTPUT_DIR, f"{target_object}_queries.json"), 'w') as f:
        json.dump(text_queries, f, indent=2)

    # 3. EXPLORATION PHASE
    # ====================
    print("\n[Phase 3/5] Running Exploration Phase...")
    dataset = COCO(config.DATASET_DIR)

    retriever = Retriever(
        dataset=dataset,
        clip_model=clip_model,
        clip_processor=clip_processor,
        dreamsim_model=dreamsim_model,
        device=device
    )
    retriever.build_index()

    vlm_filter = VLMFilter(vlm_filter_model, vlm_processor, device)
    od_filter = ObjectDetectorFilter(object_detector, object_detector_processor, device)

    exploration_candidates = []
    print(f"Retrieving images for {len(text_queries)} text queries...")
    for query in tqdm(text_queries, desc="Exploration"):
        retrieved_paths = retriever.retrieve_from_text(query, k=config.EXPLORATION_K)

        for img_path in retrieved_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                if not od_filter.is_object_present(image, target_object) and \
                        vlm_filter.is_object_present(image, target_object):
                    exploration_candidates.append(img_path)
            except Exception as e:
                print(f"Warning: Could not process image {img_path}. Error: {e}")

    exploration_candidates = sorted(list(set(exploration_candidates)))
    print(f"Found {len(exploration_candidates)} successful candidates after Exploration.")

    utils.save_image_grid(
        exploration_candidates,
        os.path.join(config.OUTPUT_DIR, f"{target_object}_exploration_results.png"),
        "Exploration Phase Results"
    )

    # 4. EXPLOITATION PHASE
    # =====================
    if not exploration_candidates:
        print("\nNo candidates found in exploration phase. Stopping pipeline.")
        return

    print("\n[Phase 4/5] Running Exploitation Phase...")
    exploitation_candidates_raw = []
    for img_path in tqdm(exploration_candidates, desc="Exploitation"):
        retrieved_paths = retriever.retrieve_from_image(img_path, k=config.EXPLOITATION_K)
        exploitation_candidates_raw.extend(retrieved_paths)

    # Remove duplicates from the initial retrieved set
    exploitation_candidates_raw = sorted(list(set(exploitation_candidates_raw)))

    print(f"Retrieved {len(exploitation_candidates_raw)} raw candidates. Now de-duplicating...")

    # NEW: De-duplication step using DreamSim as specified in the paper
    exploitation_candidates = retriever.deduplicate(exploitation_candidates_raw)

    # Final cleanup
    exploitation_candidates = sorted(list(set(exploitation_candidates) - set(exploration_candidates)))
    print(f"Found {len(exploitation_candidates)} new, unique candidates after Exploitation and De-duplication.")

    utils.save_image_grid(
        exploitation_candidates,
        os.path.join(config.OUTPUT_DIR, f"{target_object}_exploitation_results.png"),
        "Exploitation Phase Results"
    )

    # 5. CLUSTERING
    # =============
    print("\n[Phase 5/5] Clustering final candidates using DreamSim...")
    all_successful_images = sorted(list(set(exploration_candidates + exploitation_candidates)))

    if len(all_successful_images) < 2:
        print("Not enough images to perform clustering. Need at least 2.")
        return

    # MODIFIED: Clusterer now uses DreamSim
    clusterer = Clusterer(
        dreamsim_model=dreamsim_model,
        device=device
    )
    clusters = clusterer.perform_clustering(all_successful_images)

    output_clusters_path = os.path.join(config.OUTPUT_DIR, f"{target_object}_clusters.json")
    with open(output_clusters_path, 'w') as f:
        json.dump(clusters, f, indent=2)

    print(f"Clustering complete. Found {len(clusters)} clusters.")
    print(f"Results saved to {output_clusters_path}")

    utils.save_cluster_grids(
        clusters,
        all_successful_images,
        os.path.join(config.OUTPUT_DIR, f"{target_object}_clusters_visualization.png")
    )
    print(f"Cluster visualization saved.")
    print("\n✅ Pipeline finished successfully!")


if __name__ == "__main__":
    TARGET_OBJECT = "leopard"
    TARGET_VLM_MODEL = config.VLM_MODEL_NAME
    run_pipeline(TARGET_OBJECT, TARGET_VLM_MODEL)