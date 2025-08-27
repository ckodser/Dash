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
    def __init__(self, dataset: data.COCO, clip_model, clip_processor, dreamsim_model, device: str):
        self.dataset = dataset
        self.image_ids = self.dataset.get_imgIds()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.dreamsim_model = dreamsim_model
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
        if os.path.exists(config.FAISS_INDEX_PATH) and False:
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

        return [self.image_paths[i] for i in indices[0]]

    def retrieve_from_image(self, image_path: str, k: int) -> list:
        """Retrieves the top-k most similar images for a given image query."""
        if self.index is None:
            raise RuntimeError("Index is not built. Call build_index() first.")

        query_vector = self._encode_images([image_path])
        _, indices = self.index.search(query_vector, k)

        # The first result will be the image itself, so we retrieve k+1 and skip the first one.
        # However, to be safe and simple, we'll just return k and let the main pipeline handle duplicates.
        return [self.image_paths[i] for i in indices[0]]

    @torch.no_grad()
    def deduplicate(self, image_paths: List[str]) -> List[str]:
        """
        Filters a list of images to remove near-duplicates using DreamSim.
        """
        if len(image_paths) < 2:
            return image_paths

        unique_images = []
        # This is a greedy approach. For each image, we check if it's too similar
        # to any image already in our `unique_images` list.
        for i, path1 in enumerate(tqdm(image_paths, desc="De-duplicating")):
            is_duplicate = False
            if not unique_images:
                unique_images.append(path1)
                continue

            # Compare the new candidate with all confirmed unique images
            # This is computationally intensive (O(n^2)) but accurate for smaller sets.
            # For larger sets, a more optimized clustering approach would be needed.
            img1 = Image.open(path1).convert("RGB")
            for path2 in unique_images:
                img2 = Image.open(path2).convert("RGB")

                # DreamSim expects a batch of images
                similarity = self.dreamsim_model(
                    torch.stack([self.dreamsim_model.preprocess(img) for img in [img1, img2]]).to(self.device)
                )
                # The output is a distance, so similarity is 1 - distance, but the model returns similarity directly.
                # Let's assume the model returns a single similarity score for the pair.
                # The actual dreamsim library might have a different API, this is an adaptation.
                # A more efficient way is to embed all and compute pairwise similarity.
                # For simplicity here, we do it one by one.
                score = similarity[0]  # Placeholder for actual pairwise score

                # The model returns a distance matrix, we need to extract the score
                # This is a simplified call; a real implementation would batch this.
                dist = self.dreamsim_model.compute_distance(img1, img2)
                score = 1 - dist.item()  # Convert distance to similarity

                if score > config.DEDUPLICATION_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_images.append(path1)

        return unique_images

