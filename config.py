# Central configuration file for the DASH pipeline.
# This makes it easy to change models and parameters.

import torch
import os


title = "coco_test"
# --- Model Configuration ---
HF_TOKEN_PATH = "./hf_token.txt"
HF_HOME = "/fs/nexus-scratch/asoltan3/.cache"
# VLM to be tested for hallucinations (as per the paper)
VLM_MODEL_NAME = "google/paligemma-3b-mix-224"

# LLM for generating text queries (using a smaller, accessible model for this example)
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Object detector model (as per the paper)
OBJECT_DETECTOR_MODEL_NAME = "google/owlv2-base-patch16-ensemble"

# CLIP model for image/text embeddings and retrieval
CLIP_MODEL_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

DREAMSIM_MODEL_NAME = "dreamsim-openclip-ViT-H-14"

# --- Pipeline Parameters ---
# Number of initial text queries to generate
retriever_batch_size = 32

NUM_TEXT_QUERIES = 50  # Paper uses 50

# Number of images to retrieve in the exploration phase for each text query
EXPLORATION_K = 20  # Paper uses 20

# Number of images to retrieve in the exploitation phase for each successful candidate
EXPLOITATION_K = 50  # Paper uses 50

# --- Object Detector ---
# Confidence threshold. If detector confidence is > this, we assume the object is present.
# The paper uses a conservative (low) threshold of 0.1.
OD_CONFIDENCE_THRESHOLD = 0.1

# --- De-duplication (as per paper) ---
# Similarity threshold for filtering out near-duplicates.
DEDUPLICATION_THRESHOLD = 0.9

# --- Clustering (as per paper) ---
# Use the DreamSim distance threshold specified in the paper.
DREAMSIM_CLUSTER_DISTANCE_THRESHOLD = 0.6

# --- File Paths ---
# Directory to store all outputs (results, visualizations)
OUTPUT_DIR = "dash_output"

# Directory for our dummy dataset.
DATASET_DIR = "/fs/cml-datasets/coco"
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "retrieval.index")

# --- System ---
# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DASH-OPT Specific Parameters ---
# Number of optimization steps for generating each image (Paper Appendix C uses 25)
DASH_OPT_STEPS = 25

# Lambda weight for the detector loss in the optimization objective (Paper uses 1.0)
DASH_OPT_LAMBDA = 1.0

# --- DASH-OPT Generation ---
# Models for the distilled Stable Diffusion XL pipeline
SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_UNET_MODEL_ID = "latent-consistency/lcm-sdxl"

# Optimization parameters
DASH_OPT_LR = 0.1
DASH_OPT_INFERENCE_STEPS = 1
DASH_OPT_GUIDANCE_SCALE = 1.0 # !!!!!!!!!!!!! NOT SURE ABOUT THIS.
# The paper mentions thresholding the detector confidence before the loss calculation
DASH_OPT_DETECTOR_THRESHOLD = 0.05

# SDXL Latent space dimensions
DASH_OPT_LATENT_CHANNELS = 4
DASH_OPT_LATENT_HEIGHT = 128
DASH_OPT_LATENT_WIDTH = 128
