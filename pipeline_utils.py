# Helper functions for loading models, preprocessing images, and visualization.

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    PaliGemmaForConditionalGeneration, PaliGemmaProcessor,
    Owlv2ForObjectDetection, Owlv2Processor,
    CLIPModel, CLIPProcessor
)
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import math
from typing import Dict, List
from dreamsim import dreamsim
import hashlib

import config


# --- Image hashing and math ---


def image_hash(img: Image.Image) -> str:
    """Generate a hash for a PIL image for comparison."""
    # Use pixel data only (metadata ignored)
    return hashlib.md5(img.tobytes()).hexdigest()

def image_difference(list_a, list_b):
    """Return images in list_a that are not in list_b."""
    hashes_b = {image_hash(img) for img in list_b}
    return [img for img in list_a if image_hash(img) not in hashes_b]

def unique_images(images):
    """Return unique images from a list (removes duplicates)."""
    seen = set()
    unique = []
    for img in images:
        h = image_hash(img)
        if h not in seen:
            seen.add(h)
            unique.append(img)
    return unique



# --- Model Loading ---

def load_models(vlm_model_name: str, device: str) -> dict:
    """Loads all required models from Hugging Face and moves them to the specified device."""
    print("Loading all models... This might take a while.")
    with open(config.HF_TOKEN_PATH) as f:
        hf_token = f.read().strip()
    # LLM for Query Generation
    llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir=config.HF_HOME, token=hf_token)
    # Use bfloat16 for memory efficiency if available
    llm_model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        cache_dir=config.HF_HOME,
        token=hf_token,
    )

    # VLM for Filtering
    vlm = PaliGemmaForConditionalGeneration.from_pretrained(
        vlm_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        cache_dir=config.HF_HOME,
        token=hf_token
    ).to(device).eval()

    # Object Detector
    object_detector = Owlv2ForObjectDetection.from_pretrained(
        config.OBJECT_DETECTOR_MODEL_NAME,
        cache_dir=config.HF_HOME,
        token=hf_token,
    ).to(device).eval()

    # CLIP for Retrieval
    clip = CLIPModel.from_pretrained(
        config.CLIP_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        cache_dir=config.HF_HOME,
        token=hf_token,
    ).to(device).eval()
    print(f"Loading DreamSim model: {config.DREAMSIM_MODEL_NAME}...")
    dreamsim_model, dreamsim_processor = dreamsim(
        pretrained=True,
        device=device,
        cache_dir=config.HF_HOME,
        normalize_embeds=True,
        dreamsim_type="ensemble",
        use_patch_model=False
    )
    dreamsim_model.eval()

    print("All models loaded successfully.")
    return {
        "llm_generator": {"model": llm_model, "tokenizer": llm_tokenizer},
        "vlm": vlm,
        "object_detector": object_detector,
        "clip": clip,
        "dreamsim": dreamsim_model, 
        "dreamsim_processor": dreamsim_processor
    }


def load_processors() -> dict:
    """Loads all necessary model processors from Hugging Face."""
    with open(config.HF_TOKEN_PATH) as f:
        hf_token = f.read().strip()
    return {
        "vlm": PaliGemmaProcessor.from_pretrained(config.VLM_MODEL_NAME, cache_dir=config.HF_HOME, token=hf_token),
        "object_detector": Owlv2Processor.from_pretrained(config.OBJECT_DETECTOR_MODEL_NAME, cache_dir=config.HF_HOME,
                                                           token=hf_token),
        "clip": CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME, cache_dir=config.HF_HOME, token=hf_token)
    }


# --- Visualization ---

def save_images(images: List[Image], target_object: str, title: str):
    """Creates and saves a group of images."""
    images_with_path = []
    output_dir = os.path.join(config.OUTPUT_DIR, f"{title}_{target_object}")
    os.makedirs(output_dir, exist_ok=True)
    for i,image in enumerate(images):
        image.save(os.path.join(output_dir, f"image_{i}.png"))
        images_with_path.append((image, os.path.join(output_dir, f"image_{i}.png")))
    print(f"Saved images to {output_dir}")
    return images_with_path


def save_cluster_grids(clusters: Dict[int, List[int]], all_image_paths: List[str], output_path: str):
    """Saves a visualization of all clusters as a single large image."""
    if not clusters:
        print("No clusters to visualize.")
        return

    cluster_images = []
    max_images_per_cluster_vis = 10  # Limit images shown per cluster for readability

    for cluster_id, image_indices in sorted(clusters.items()):
        cluster_title = f"Cluster {cluster_id} ({len(image_indices)} images)"
        paths = [all_image_paths[i] for i in image_indices[:max_images_per_cluster_vis]]

        # Create a small grid for each cluster
        num_images = len(paths)
        if num_images == 0: continue

        cell_size = 128
        header_height = 30

        # Make the grid horizontal
        grid_cols = num_images
        grid_rows = 1

        grid_width = grid_cols * cell_size
        grid_height = grid_rows * cell_size + header_height

        grid = Image.new('RGB', (grid_width, grid_height), 'lightgray')
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        draw.text((5, 5), cluster_title, fill="black", font=font)

        for i, path in enumerate(paths):
            try:
                img = Image.open(path).resize((cell_size, cell_size))
                grid.paste(img, (i * cell_size, header_height))
            except Exception as e:
                print(f"Could not load image {path} for cluster grid: {e}")

        cluster_images.append(grid)

    if not cluster_images:
        print("Failed to generate any cluster images.")
        return

    # Combine all cluster grids into one tall image
    total_width = max(img.width for img in cluster_images)
    total_height = sum(img.height for img in cluster_images)

    final_image = Image.new('RGB', (total_width, total_height), 'white')
    current_y = 0
    for img in cluster_images:
        final_image.paste(img, (0, current_y))
        current_y += img.height

    final_image.save(output_path)
