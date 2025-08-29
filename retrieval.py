# Handles image retrieval from a dataset using a vector index.
# This is used in both the Exploration and Exploitation phases.

import faiss
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torch
from typing import List

import config
import data


class Retriever:
    def __init__(self, dataset: data.COCO, clip_model, clip_processor, dreamsim_model, dreamsim_processor, device: str):
        self.dataset = dataset
        self.image_ids = self.dataset.get_imgIds()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.dreamsim_model = dreamsim_model
        self.dreamsim_processor = dreamsim_processor
        self.device = device
        self.index = None

    @torch.no_grad()
    def _encode_images(self, images: list) -> np.ndarray:
        """Encodes a batch of images using the CLIP model."""
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)

        # Handle potential dtype issues
        if self.device == "cuda":
            inputs = {k: v.to(torch.float16) for k, v in inputs.items()}

        image_features = self.clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype('float32')

    @torch.no_grad()
    def _encode_text(self, text: str) -> np.ndarray:
        """Encodes a text string using the CLIP model."""
        inputs = self.clip_processor(text=[text], return_tensors="pt").to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype('float32')

    def build_index(self):
        """
        Builds a FAISS index for all images in the directory.
        If an index file already exists, it loads it.
        """
        if os.path.exists(config.FAISS_INDEX_PATH):
            print(f"Loading existing FAISS index from {config.FAISS_INDEX_PATH}")
            self.index = faiss.read_index(config.FAISS_INDEX_PATH)
            assert self.index.ntotal == len(self.dataset), "Index size mismatch with image count!"
            return

        print("Building new FAISS index for the image dataset...")
        all_features = []
        batch_size = config.retriever_batch_size
        for i in tqdm(range(0, len(self.image_ids), batch_size), desc="Encoding Images"):
            batch_ids = self.image_ids[i:i + batch_size]
            batch_images = [self.dataset[id][0] for id in batch_ids]
            batch_features = self._encode_images(batch_images)
            all_features.append(batch_features)
            
        all_features = np.vstack(all_features)

        dimension = all_features.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Using Inner Product (cosine similarity)
        self.index.add(all_features)

        faiss.write_index(self.index, config.FAISS_INDEX_PATH)
        print(f"FAISS index built and saved to {config.FAISS_INDEX_PATH}")

    def retrieve_from_text(self, query_text: str, k: int) -> list:
        """Retrieves the top-k most similar images for a given text query."""
        if self.index is None:
            raise RuntimeError("Index is not built. Call build_index() first.")

        query_vector = self._encode_text(query_text)
        _, indices = self.index.search(query_vector, k)
        selected_image_ids = [self.image_ids[image_id] for image_id in indices[0]]
        return [self.dataset[i][0] for i in selected_image_ids]

    def retrieve_from_image(self, image: Image, k: int) -> list:
        """Retrieves the top-k most similar images for a given image query."""
        if self.index is None:
            raise RuntimeError("Index is not built. Call build_index() first.")

        query_vector = self._encode_images([image])
        _, indices = self.index.search(query_vector, k)
        selected_image_ids = [self.image_ids[i] for i in indices[0]]
        return [self.dataset[image_id][0] for image_id in selected_image_ids]

    @torch.no_grad()
    def get_embeds(self, images: List[Image]):
        unique_embeddings_tensor = None

        for i in tqdm(range(0, len(images)), desc="Computing embeddings"):
            image_tensor = self.dreamsim_processor(images[i]).to(self.device)
            current_embedding = self.dreamsim_model.embed(image_tensor)
            current_embedding = current_embedding/torch.linalg.norm(current_embedding)
            if unique_embeddings_tensor is None:
                unique_embeddings_tensor = current_embedding
            else:
                unique_embeddings_tensor = torch.cat([unique_embeddings_tensor, current_embedding], dim=0)
                    
        return unique_embeddings_tensor

        
    @torch.no_grad()
    def deduplicate(self, images: List[Image]):
        """
        Filters a list of images to remove near-duplicates using DreamSim.
        """
        if len(images) < 2:
            return images, self.get_embeds(images)

        unique_indices = []
        unique_embeddings_tensor = None

        for i in tqdm(range(0, len(images)), desc="Computing embeddings"):
            image_tensor = self.dreamsim_processor(images[i]).to(self.device)
            current_embedding = self.dreamsim_model.embed(image_tensor)
            current_embedding = current_embedding/torch.linalg.norm(current_embedding)
            if unique_embeddings_tensor is None:
                unique_embeddings_tensor = current_embedding
                unique_indices.append(i)
            else:
                distances = unique_embeddings_tensor@current_embedding.T
                if distances.max() <= config.DEDUPLICATION_THRESHOLD:
                    unique_indices.append(i)
                    # Add the new unique embedding for future comparisons
                    unique_embeddings_tensor = torch.cat([unique_embeddings_tensor, current_embedding], dim=0)
        unique_images = [images[i] for i in unique_indices]

        print(f"Original images: {len(images)}. Unique images: {len(unique_images)}.")
        return unique_images, unique_embeddings_tensor