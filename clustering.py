# Performs the final clustering of successful hallucination images.

import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from PIL import Image
from typing import List, Dict
from tqdm import tqdm
import itertools

import config

class Clusterer:
    def __init__(self, dreamsim_model, device):
        self.dreamsim_model = dreamsim_model
        self.device = device

    @torch.no_grad()
    def get_distance_matrix(self, image_paths: List[str]) -> np.ndarray:
        """
        Computes the pairwise DreamSim distance matrix for a list of images.
        """
        num_images = len(image_paths)
        distance_matrix = np.zeros((num_images, num_images))

        # Pre-load and preprocess all images
        images = [Image.open(path).convert("RGB") for path in image_paths]

        # This is memory intensive but much faster than one-by-one comparisons
        try:
            embeddings = self.dreamsim_model.embed(images)
            # Compute pairwise cosine distance: dist(u, v) = 1 - (u • v) / (||u|| * ||v||)
            # Since embeddings are normalized, this is 1 - (u @ v.T)
            similarity_matrix = embeddings @ embeddings.T
            distance_matrix = 1 - similarity_matrix.cpu().numpy()
        except Exception as e:
            # Fallback to one-by-one if batching fails (e.g., OOM)
            print(f"Batch embedding failed ({e}), falling back to pairwise comparison.")
            for i, j in tqdm(list(itertools.combinations(range(num_images), 2)), desc="Computing Distance Matrix"):
                dist = self.dreamsim_model.compute_distance(images[i], images[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist.item()

        return distance_matrix

    def perform_clustering(self, image_paths: List[str]) -> Dict[int, List[int]]:
        """
        Performs agglomerative clustering using a precomputed DreamSim distance matrix.
        """
        if not image_paths or len(image_paths) < 2:
            return {}

        print("Computing DreamSim distance matrix for clustering...")
        distance_matrix = self.get_distance_matrix(image_paths)

        # MODIFIED: Use 'precomputed' metric as we provide the distance matrix directly.
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',  # Use our custom distance matrix
            linkage='average',
            distance_threshold=config.DREAMSIM_CLUSTER_DISTANCE_THRESHOLD
        )

        # The model expects a condensed distance matrix for 'precomputed', but we can fit the full one.
        labels = clustering_model.fit_predict(distance_matrix)

        clusters = {}
        for i, label in enumerate(labels):
            if int(label) not in clusters:
                clusters[int(label)] = []
            clusters[int(label)].append(i)

        return clusters
