# Central configuration file for the DASH pipeline.
# This makes it easy to change models and parameters.

import torch
import os


# --- Model Configuration ---
HF_TOKEN_PATH = "./hf_token.txt"
HF_HOME = "/cmlscratch/asoltan3/.cache"
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
