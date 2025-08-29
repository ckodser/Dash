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
    def __init__(self, dreamsim_model, dreamsim_processor, device):
        self.dreamsim_model = dreamsim_model
        self.device = device
        self.dreamsim_processor= dreamsim_processor

    @torch.no_grad()
    def get_distance_matrix(self, embeddings) -> np.ndarray:
        """
        Computes the pairwise DreamSim distance matrix for a list of images.
        """
        num_images = embeddings.shape[0]
        distance_matrix = np.zeros((num_images, num_images))

        similarity_matrix = embeddings @ embeddings.T
        distance_matrix = 1 - similarity_matrix.cpu().numpy()

        return distance_matrix

    def perform_clustering(self, images_with_path: List, embed) -> Dict[int, List[int]]:
        """
        Performs agglomerative clustering using a precomputed DreamSim distance matrix.
        """
        if not images_with_path or len(images_with_path) < 2:
            return {}

        print("Computing DreamSim distance matrix for clustering...")
        distance_matrix = self.get_distance_matrix(embed)

        
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed', 
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
