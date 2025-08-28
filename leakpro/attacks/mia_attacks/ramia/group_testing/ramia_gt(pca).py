"""Implementation of the RaMIA attack using RMIA as the base membership inference method."""

from typing import Literal, Callable, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

import optuna
import os
import torch
import pickle
import torchvision.transforms as transforms
from torch import tensor, float32, cat

from leakpro.schemas import OptunaConfig, avg_tpr_at_low_fpr
from leakpro.input_handler.modality_extensions.image_extension import ImageExtension
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.boosting import Memorization
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.attacks.utils.utils import softmax_logits
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.metrics.attack_result import MIAResult
from leakpro.signals.signal import ModelRescaledLogits, ModelLogits
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger

from leakpro.attacks.mia_attacks.ramia.group_testing.bridge_files.fedgt_bridge import GroupTestDecoder
from leakpro.attacks.mia_attacks.ramia.group_testing.bridge_files.fedqgt_bridge import QGTDecoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score

# Define the TransformedDataset class at module level so it can be pickled
class TransformedDataset:
    """Dataset class for transformed samples that mirrors CifarDataset structure."""
    
    def __init__(self, x, y, transform=None, metadata=None):
        """
        Initialize with samples and labels.
        
        Args:
            x: Tensor of input images
            y: Tensor of labels
            transform: Optional transform to be applied
            metadata: Optional metadata dictionary
        """
        self.x = x  # Data tensor
        self.y = y  # Labels tensor
        self.transform = transform
        self.metadata = metadata if metadata is not None else {
            "original_splits": [],
            "original_idx": [],
            "raw_data": None
        }
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.y)
    
    def __getitem__(self, idx):
        """Retrieve the sample and its label at index 'idx'."""
        image = self.x[idx]
        label = self.y[idx]
        
        # Apply transformations to the image if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def subset(self, indices):
        """Return a subset of the dataset based on the given indices."""
        subset_metadata = {
            "original_splits": [self.metadata.get("original_splits", [])[i] for i in indices] if self.metadata.get("original_splits") else [],
            "original_idx": [self.metadata.get("original_idx", [])[i] for i in indices] if self.metadata.get("original_idx") else [],
            "raw_data": self.metadata.get("raw_data", None)
        }
        
        # Preserve RaMIA-specific metadata at the top level
        for key in self.metadata:
            if key not in subset_metadata and key not in ["original_splits", "original_idx", "raw_data"]:
                subset_metadata[key] = self.metadata[key]
        
        return TransformedDataset(
            x=self.x[indices],
            y=self.y[indices],
            transform=self.transform,
            metadata=subset_metadata
        )
    
class AttackRaMIA_GT(AbstractMIA):
    """Implementation of the RaMIA attack using RMIA as the base membership inference method."""

    class AttackConfig(BaseModel):
        """Configuration for the RaMIA attack."""

        # RaMIA attack parameters
        num_transforms: int = Field(default=15, ge=1, le=200, description="Number of transformations to apply to each sample in a range") # Range transformation parameters
        # range_num_audit_samples: int = Field(default=50, ge=5, description="Number of audit samples to use for range membership inference") # Range transformation parameters
        dataset_type: Literal["image", "tabular", "text"] = Field(default="image", description="Type of dataset being analyzed") # Dataset type configuration
        transform_type: Literal["standard", "cifar", "custom"] = Field(default="cifar", description="Type of transformation to apply to images") # Image transformation configuration
        optuna_config: OptunaConfig = Field(default=OptunaConfig()) # Optuna configuration for hyperparameter search
        objective: Optional[Callable[[MIAResult], float]] = Field(default=None, description="Objective function for optimization")
        
        # Trimming parameters
        trim_lower_percentile: int = Field(default=20, ge=0, le=100, description="Lower percentile for trimming (for synthetic data)")
        trim_auto_tune: bool = Field(default=True, description="Whether to automatically tune trimming parameters")
        
        # RMIA specific parameters
        temperature: float = Field(default=2.0, ge=0.0, description="Softmax temperature for RMIA")
        attack_data_fraction: float = Field(default=0.1, ge=0.0, le=1.0, description="Part of available attack data to use for RMIA attack")
        gamma: float = Field(default=2.0, ge=0.0, description="Parameter to threshold LLRs for RMIA", 
                             json_schema_extra = {"optuna": {"type": "float", "low": 0.1, "high": 10, "log": True}})
        offline_a: float = Field(default=0.33, ge=0.0, le=1.0, description="Parameter to estimate the marginal p(x) for RMIA",
                                json_schema_extra = {"optuna": {"type": "float", "low": 0.0, "high": 1.0,"enabled_if": lambda model: not model.online}})

        # Common parameters for MIA
        num_shadow_models: int = Field(default=4, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.6, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")
        online: bool = Field(default=False, description="Online vs offline attack") 
        eval_batch_size: int = Field(default=32, ge=1, description="Batch size for evaluation")
        
        # Memorization boosting
        memorization: bool = Field(default=False, description="Activate memorization boosting")
        use_privacy_score: bool = Field(default=False, description="Filter based on privacy score as well as memorization score")
        memorization_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Set percentile for most vulnerable data points")
        min_num_memorization_audit_points: int = Field(default=10, ge=1, description="Set minimum allowed audit points after memorization")
        num_memorization_audit_points: int = Field(default=0, ge=0, description="Directly set number of most vulnerable audit data points")

        #group testing
        pca_components: int = Field(default=10, ge=1, le=100, description="Max number of PCA components")

        @model_validator(mode="after")
        def check_num_shadow_models_if_online(self) -> Self:
            """Check if the number of shadow models is at least 2 when online is True."""
            if self.online and self.num_shadow_models < 2:
                raise ValueError("When online is True, num_shadow_models must be >= 2")
            return self
        
        @model_validator(mode='after')
        def handle_objective(self) -> Self:
            """Set default objective if not provided."""
            if self.objective is None:
                self.objective = avg_tpr_at_low_fpr
            return self

    def __init__(self:Self,
                 handler: MIAHandler,
                 configs: dict
                 ) -> None:
        """Initialize the RaMIA attack.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.
        """
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)
        self.gt_decoder = GroupTestDecoder()  # NEW: Decoder instance
        self.qgt_decoder = QGTDecoder()
        self.parity_matrices = {}  # NEW: {sample_idx: H_matrix}

        # Initializes the parent metric
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.online is False and self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow models.")

        self.shadow_models = []
        
        # Use ModelLogits signal for RMIA instead of ModelRescaledLogits used in LiRA
        self.signal = ModelLogits()
        self.epsilon = 1e-6  # Small value to avoid division by zero
        self.shadow_models = None
        self.shadow_model_indices = None
        
        # Initialize image extension for transformations if using image data
        if self.dataset_type == "image":
            self.image_extension = ImageExtension(handler)
            transform_config = configs.get("transform_config", {})
            self.range_transforms = self.image_extension.get_transform(self.transform_type, **transform_config)
        else:
            # For non-image data, we'll initialize transforms in prepare_attack
            self.range_transforms = None
            logger.warning(f"Using dataset type '{self.dataset_type}', custom transformations may be needed")

        # Folder to store intermediate results for RMIA
        self.attack_folder_path = "leakpro_output/attacks/ramia_rmia"
        os.makedirs(self.attack_folder_path, exist_ok=True)
        self.load_for_optuna = False
        
        # Set default objective function if not provided
        if self.configs.objective is None:
            self.objective = avg_tpr_at_low_fpr
        else:
            self.objective = self.configs.objective

    def description(self: Self) -> dict:
        """Return a description of the attack for documentation and reporting.
        
        Returns:
            dict: Contains title, reference, summary, and detailed description
        """
        title_str = "Range Membership Inference Attack with RMIA"
        
        reference_str = "Tao J, Shokri R. Range Membership Inference Attacks"
        
        summary_str = ("RaMIA extends membership inference to test if a range of data points contains "
                       "any training samples, providing more comprehensive privacy auditing. "
                       "This implementation uses RMIA as the base membership inference method.")
        
        detailed_str = (
            "Range Membership Inference Attack (RaMIA) with RMIA combines two powerful approaches: "
            "1) RaMIA extends privacy auditing beyond exact matches to consider similar data points, and "
            "2) RMIA (Relative Membership Inference Attack) uses likelihood ratios to determine membership. "
            "Together, these methods provide a more comprehensive and effective privacy auditing framework "
            "that can detect privacy leakage from transformed or similar versions of training data."
        )
        
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str
        }
    
    def create_transformed_dataset(self: Self, num_samples: int) -> tuple:
        """Generate transformed ranges for selected audit samples, balanced between IN and OUT members.
        
        Args:
            num_samples: Number of audit samples to select for range creation
            
        Returns:
            Tuple containing:
                - transformed_samples: List of lists of transformed samples
                - true_labels: List of binary labels (1=IN, 0=OUT)
                - selected_indices: Indices from original audit dataset that were selected
        """
        self.selected_indices = np.concatenate([self.in_members, self.out_members])

        # Validation checks
        unique_indices = np.unique(self.selected_indices)
        if len(unique_indices) != len(self.selected_indices):
            raise ValueError("Duplicate indices selected in audit samples")

        # Get dataset from handler
        dataset = self.handler.population

        transformed_samples = []
        true_labels = []

        # Process each selected index
        for idx in tqdm(self.selected_indices, desc="Creating transformed ranges"):
            data_idx = self.audit_data_indices[idx]
            
            # Get image using the method from ImageExtension
            raw_sample = self.image_extension.get_data(dataset, data_idx)
            
            # Skip if we couldn't get the data
            if raw_sample is None:
                logger.warning(f"Failed to get data for index {data_idx}")
                continue
                
            # Convert to PIL image using the existing method
            sample = self.image_extension.to_pil_image(raw_sample)
            
            # Skip if conversion failed
            if sample is None:
                logger.warning(f"Failed to convert sample to PIL image for index {data_idx}")
                continue

            # Generate transformed samples
            try:
                range_samples = []
                for transform_idx in range(self.num_transforms):
                    # Apply transformations to the image
                    transformed = self.range_transforms(sample, transform_idx)
                    range_samples.append(transformed)

                logger.info(f"Sample {idx}: Created {len(range_samples)} transformed samples")

                # Perform group testing for those transformed samples
                representatives, H = self.perform_group_testing(range_samples) 
                # representatives, H = self.perform_vgg_group_testing(range_samples)

                # Store parity matrix and representatives
                self.parity_matrices[idx] = H  # Now stores group info for representatives     
                transformed_samples.append(representatives)
                # transformed_samples.append(range_samples)
                true_labels.append(1 if idx in self.in_members else 0)
                # logger.info(f"Successfully created transforms for index {idx}")
                logger.info(f"Created {len(representatives)} representatives for sample {idx}")
            except Exception as e:
                logger.warning(f"Error transforming sample {idx}: {e}")
                continue

        logger.info(f"Created {len(transformed_samples)} ranges "
                    f"({sum(true_labels)} IN, {len(true_labels)-sum(true_labels)} OUT)")
        
        return transformed_samples, true_labels, self.selected_indices
    
    # Medoid approach but clustering strategy is different 
    # def perform_group_testing(self: Self, range_samples: list, n_clusters=10, m=4) -> list:
    #     """Perform group testing on the transformed samples.

    #     Args:
    #         range_samples: List of transformed samples
    #         n_clusters: Number of clusters to form
    #         n_representatives: Number of representative samples to select

    #     Returns:
    #         List of refined samples after group testing
    #     """
    #     # Step 1: Convert to PCA representation
    #     # Convert tensors to numpy arrays and flatten
    #     flat_features = [sample.numpy().flatten() if isinstance(sample, torch.Tensor) 
    #                     else np.array(sample).flatten() 
    #                     for sample in range_samples]
    
        
    #     # Determine optimal PCA components (min of samples or 10)
    #     n_components = min(len(range_samples)-1, self.configs.pca_components)
    #     pca = PCA(n_components=n_components)
    #     pca_features = pca.fit_transform(flat_features)

    #     logger.info(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")

    #     # Step 2: Apply k-means clustering
    #     n_clusters = min(n_clusters, len(pca_features)-1)
    #     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=1234, max_iter=500, tol=1e-4)
    #     cluster_labels = kmeans.fit_predict(pca_features)
    #     centroids = kmeans.cluster_centers_

    #     logger.info(f"Created {n_clusters} clusters with centers at:")
    #     for i, centroid in enumerate(centroids):
    #         logger.info(f"  Cluster {i}: center at {np.round(centroid[:3], 2)}... ({len(np.where(cluster_labels == i)[0])} samples)")
            
    #     # Step 3: Build assignment matrix with top-m closest centroids per sample
    #     distances = euclidean_distances(pca_features, centroids)
    #     top_m_centroids = np.argsort(distances, axis=1)[:, :m]
        
    #     assignment_matrix = np.zeros((len(range_samples), n_clusters), dtype=int)
    #     for i in range(len(range_samples)):
    #         assignment_matrix[i, top_m_centroids[i]] = 1


    #     # Step 4: Compute new centroids based on assignment matrix
    #     # new_centroids = []
    #     representative_indices = []
    #     for j in range(n_clusters):
    #         samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
    #         if len(samples_in_cluster) > 0:
    #             if len(samples_in_cluster) > 1:
    #                 # Calculate pairwise distances within this cluster
    #                 cluster_features = pca_features[samples_in_cluster]
    #                 within_distances = euclidean_distances(cluster_features)
                    
    #                 # Sum distances for each sample
    #                 sum_distances = np.sum(within_distances, axis=1)
                    
    #                 # Find the sample with minimum total distance (the medoid)
    #                 medoid_idx_in_cluster = np.argmin(sum_distances)
    #                 representative_idx = samples_in_cluster[medoid_idx_in_cluster]
    #             else:
    #                 # If only one sample, it's automatically the medoid
    #                 representative_idx = samples_in_cluster[0]
                    
    #             representative_indices.append(representative_idx)
    #         else:
    #             logger.warning(f"Group {j} has no samples assigned to it!")
            
        
    #     # Return samples at representative indices
    #     representative_samples = [range_samples[idx] for idx in representative_indices]
        
    #     # Build group testing matrix H using the precomputed assignment_matrix
    #     # Get the cluster assignments for representatives and transpose
    #     H = assignment_matrix

    #     return representative_samples, H

    #     # rep_assignment_matrix = np.zeros((len(representative_indices), n_clusters), dtype=int)
        
    #     # for i, rep_idx in enumerate(representative_indices):
    #     #     rep_assignment_matrix[i] = assignment_matrix[rep_idx]
        
    #     # # Transpose to get groups × representatives 
    #     # H = rep_assignment_matrix.T  # Shape: (n_clusters × n_representatives)
        
    #     # logger.info(f"Created H matrix with shape {H.shape} (groups × representatives)")
        
    #     # return representative_samples, H

    # Medoid approach but clustering strategy is balanced 
    # def perform_group_testing(self, range_samples: list, n_clusters=5, m=3):
    #     """Improved group testing with balanced overlapping clusters"""
    #     # Step 1: PCA (unchanged from your code)
    #     flat_features = [sample.numpy().flatten() if isinstance(sample, torch.Tensor) 
    #                     else np.array(sample).flatten() 
    #                     for sample in range_samples]
        
    #     n_components = min(len(range_samples)-1, self.configs.pca_components)
    #     pca = PCA(n_components=n_components)
    #     pca_features = pca.fit_transform(flat_features)
        
    #     # Step 2: Run k-means to get cluster centers
    #     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20)
    #     kmeans.fit(pca_features)
    #     centroids = kmeans.cluster_centers_
        
    #     # Step 3: Calculate distances to all centroids for each sample
    #     distances = euclidean_distances(pca_features, centroids)
        
    #     # Step 4: Create balanced assignment matrix
    #     n_samples = len(range_samples)
        
    #     # Set maximum samples per cluster (for balance)
    #     max_per_cluster = int(np.ceil((n_samples * m / n_clusters) * 1.5))
        
    #     # Initialize assignment matrix
    #     assignment_matrix = np.zeros((n_samples, n_clusters), dtype=np.uint8)
        
    #     # Track how many samples are in each cluster
    #     cluster_counts = np.zeros(n_clusters, dtype=int)
    #     sample_cluster_counts = np.zeros(n_samples, dtype=int)  # Tracks how many clusters each sample belongs to
        
    #     # First pass: Assign each sample to its m closest clusters
    #     for i in range(n_samples):
    #         # Get m closest clusters
    #         closest_clusters = np.argsort(distances[i])[:m]
            
    #         # Assign to these clusters
    #         for j in closest_clusters:
    #             assignment_matrix[i, j] = 1
    #             cluster_counts[j] += 1
    #             sample_cluster_counts[i] += 1  # Track per-sample count
        
    #     # Second pass: Balance clusters that exceed maximum size
    #     for j in range(n_clusters):
    #         if cluster_counts[j] > max_per_cluster:
    #             # Sort samples by distance to this cluster
    #             samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
    #             sample_distances = distances[samples_in_cluster, j]
    #             sorted_indices = samples_in_cluster[np.argsort(sample_distances)]
                
    #             # Keep only max_per_cluster closest samples
    #             samples_to_remove = sorted_indices[max_per_cluster:]
                
    #             # Remove excess samples
    #             for i in samples_to_remove:
    #                 assignment_matrix[i, j] = 0
    #                 cluster_counts[j] -= 1
    #                 sample_cluster_counts[i] -= 1  # Decrease count
                    
    #                 # Ensure every sample still belongs to at least one cluster
    #                 # if np.sum(assignment_matrix[i]) == 0:
    #                 # Reassign to new clusters to maintain m=5
    #                 while sample_cluster_counts[i] < m:
    #                     found_cluster = False
    #                     # Find next closest cluster that's not full
    #                     for next_j in np.argsort(distances[i]):
    #                         if cluster_counts[next_j] < max_per_cluster and assignment_matrix[i, next_j] == 0:  # Added check that we haven't already assigned this sample to this cluster
    #                             assignment_matrix[i, next_j] = 1
    #                             cluster_counts[next_j] += 1
    #                             sample_cluster_counts[i] += 1
    #                             found_cluster = True  # Set flag to true when we find a cluster
    #                             break
                        
    #                     # Break if no available clusters were found
    #                     if not found_cluster:
    #                         logger.warning(f"Could not find enough available clusters for sample {i}. "
    #                                     f"Assigned to {sample_cluster_counts[i]} clusters instead of {m}.")
    #                         break  # Exit the while loop if we can't find a suitable cluster

    #     # After balancing, verify all samples have exactly m clusters
    #     for i in range(n_samples):
    #         assert np.sum(assignment_matrix[i]) == m, f"Sample {i} has {np.sum(assignment_matrix[i])} clusters (expected {m})"

        
    #     # NEW LOGGING 1: Assignment Matrix Structure
    #     logger.info(f"\nAssignment Matrix (Groups x Samples) - m={m} overlapping:")
    #     logger.info("Assignment Matrix Structure:")
    #     for j in range(n_clusters):
    #         row = assignment_matrix[:, j].tolist()  # Get samples for this group
    #         logger.info(f"Group {j:2d}: {row}")
        
    #     # NEW LOGGING 2: Sample-to-Group Mappings
    #     logger.info(f"\nSample-to-Group Assignments (verifying m={m} overlapping):")
    #     for i in range(n_samples):
    #         assigned_groups = np.where(assignment_matrix[i, :] == 1)[0].tolist()
    #         logger.info(f"Sample {i:2d}: assigned to groups {assigned_groups} (count: {len(assigned_groups)})")
        
        
        
    #     # Original logging (kept unchanged)
    #     logger.info(f"\nFinal clusters after balancing (max per cluster: {max_per_cluster}):")
    #     for j in range(n_clusters):
    #         # Get cluster members
    #         samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
    #         current_count = len(samples_in_cluster)
            
    #         # Calculate cluster center based on actual members
    #         if current_count > 0:
    #             cluster_center = np.mean(pca_features[samples_in_cluster], axis=0)
    #             logger.info(f"Cluster {j:2d}: center={np.round(cluster_center[:3], 2)} | "
    #                     f"size={current_count:2d} | "
    #                     f"overlap={current_count/m:.1f}x | "
    #                     f"{'FULL' if current_count == max_per_cluster else 'OK'}")
    #         else:
    #             logger.info(f"Cluster {j:2d}: EMPTY (no samples assigned)")
        
    #     # Step 5: Find representatives using medoids
    #     representative_indices = []
    #     cluster_to_representative = {}  # Track which sample represents each cluster
    #     for j in range(n_clusters):
    #         samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
            
    #         if len(samples_in_cluster) > 0:
    #             if len(samples_in_cluster) > 1:
    #                 # Find medoid
    #                 cluster_features = pca_features[samples_in_cluster]
    #                 within_distances = euclidean_distances(cluster_features)
    #                 sum_distances = np.sum(within_distances, axis=1)
    #                 medoid_idx = np.argmin(sum_distances)
    #                 representative_idx = samples_in_cluster[medoid_idx]
    #             else:
    #                 representative_idx = samples_in_cluster[0]
                    
    #             representative_indices.append(representative_idx)
    #             cluster_to_representative[j] = representative_idx  # Store mapping

    #     # NEW LOGGING: Log which sample represents each cluster
    #     logger.info("\nCluster Representatives (which sample represents each cluster):")
    #     for cluster_id, sample_id in sorted(cluster_to_representative.items()):
    #         logger.info(f"Cluster {cluster_id}: Represented by Sample {sample_id}")

    #     # # Log representative distribution across clusters
    #     # representative_distribution = np.sum(assignment_matrix[representative_indices], axis=0)
    #     # logger.info("\nRepresentative distribution across clusters:")
    #     # for j in range(n_clusters):
    #     #     if representative_distribution[j] > 0:
    #     #         logger.info(f"Cluster {j}: {representative_distribution[j]} representative(s)")
        
    #     # Get representative samples
    #     representative_samples = [range_samples[idx] for idx in representative_indices]
        
    #     # Return representatives and assignment matrix
    #     return representative_samples, assignment_matrix.T

    # Centroid approach
    # def perform_group_testing(self, range_samples: list, n_clusters=5, m=3):
    #     """Improved group testing with balanced overlapping clusters"""
    #     # Step 1: PCA (unchanged from your code)
    #     flat_features = [sample.numpy().flatten() if isinstance(sample, torch.Tensor) 
    #                     else np.array(sample).flatten() 
    #                     for sample in range_samples]

    #     # NEW: Log sample features before PCA
    #     logger.info("\nOriginal samples:")
    #     for i, features in enumerate(flat_features):
    #         # Show full feature vector
    #         logger.info(f"Sample {i}: {np.round(features, 2)}")
        
    #     n_components = min(len(range_samples)-1, self.configs.pca_components)
    #     pca = PCA(n_components=n_components)
    #     pca_features = pca.fit_transform(flat_features)

    #     # NEW: Log PCA features
    #     logger.info("\nSamples after PCA:")
    #     for i, features in enumerate(pca_features):
    #         # Show full PCA feature vector
    #         logger.info(f"Sample {i}: {np.round(features, 2)}")
        
    #     logger.info(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")
        
    #     # Step 2: Run k-means to get cluster centers
    #     kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20)
    #     kmeans.fit(pca_features)
    #     centroids = kmeans.cluster_centers_

    #     # NEW: Log centroids with nearby samples
    #     logger.info("\nCentroids with nearby samples:")
    #     for j, centroid in enumerate(centroids):
    #         # Show full centroid vector
            
    #         # Find samples closest to this centroid
    #         centroid_reshaped = centroid.reshape(1, -1)
    #         distances_to_centroid = euclidean_distances(pca_features, centroid_reshaped).flatten()
    #         closest_samples = np.argsort(distances_to_centroid)[:3]  # Get 3 closest samples
            
    #         logger.info(f"Centroid {j}: {np.round(centroid, 2)}  # Near samples {', '.join(map(str, closest_samples))}")
        
    #     # Step 3: Calculate distances to all centroids for each sample
    #     distances = euclidean_distances(pca_features, centroids)

    #     # NEW: Log distance matrix
    #     logger.info("\nDistances:")
    #     # Create header
    #     header = "             "
    #     for j in range(n_clusters):
    #         header += f"Centroid {j}  "
    #     logger.info(header)
        
    #     # Log each sample's distances
    #     for i, sample_distances in enumerate(distances):
    #         row = f"Sample {i}:    "
    #         for d in sample_distances:
    #             row += f"{d:.2f}       "
    #         logger.info(row)
        
    #     # Step 4: Create balanced assignment matrix
    #     n_samples = len(range_samples)
        
    #     # Set maximum samples per cluster (for balance)
    #     max_per_cluster = int(np.ceil((n_samples * m / n_clusters) * 1.5))
        
    #     # Initialize assignment matrix
    #     assignment_matrix = np.zeros((n_samples, n_clusters), dtype=np.uint8)
        
    #     # Track how many samples are in each cluster
    #     cluster_counts = np.zeros(n_clusters, dtype=int)
    #     sample_cluster_counts = np.zeros(n_samples, dtype=int)  # Tracks how many clusters each sample belongs to
        
    #     # First pass: Assign each sample to its m closest clusters
    #     for i in range(n_samples):
    #         # Get m closest clusters
    #         closest_clusters = np.argsort(distances[i])[:m]
            
    #         # Assign to these clusters
    #         for j in closest_clusters:
    #             assignment_matrix[i, j] = 1
    #             cluster_counts[j] += 1
    #             sample_cluster_counts[i] += 1  # Track per-sample count
    

    #     # NEW FORMATTED MATRIX LOG - add this new section
    #     logger.info("\nAssignment Matrix (visually formatted):")
    #     # Create header
    #     header = "           "
    #     for j in range(n_clusters):
    #         header += f"Cluster {j}  "
    #     logger.info(header)
        
    #     # Create each row
    #     for i in range(n_samples):
    #         row = f"Sample {i}:     "
    #         for j in range(n_clusters):
    #             row += f"{assignment_matrix[i,j]}         "
    #         logger.info(row)
        
    #     # NEW LOGGING 1: Assignment Matrix Structure
    #     logger.info(f"\nAssignment Matrix (Groups x Samples) - m={m} overlapping:")
    #     logger.info("Assignment Matrix Structure:")
    #     for j in range(n_clusters):
    #         row = assignment_matrix[:, j].tolist()  # Get samples for this group
    #         logger.info(f"Group {j:2d}: {row}")
        
    #     # NEW LOGGING 2: Sample-to-Group Mappings
    #     logger.info(f"\nSample-to-Group Assignments (verifying m={m} overlapping):")
    #     for i in range(n_samples):
    #         assigned_groups = np.where(assignment_matrix[i, :] == 1)[0].tolist()
    #         logger.info(f"Sample {i:2d}: assigned to groups {assigned_groups} (count: {len(assigned_groups)})")
        
        
        
    #     # Original logging (kept unchanged)
    #     logger.info(f"\nFinal clusters after balancing (max per cluster: {max_per_cluster}):")
    #     for j in range(n_clusters):
    #         # Get cluster members
    #         samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
    #         current_count = len(samples_in_cluster)
            
    #         # Calculate cluster center based on actual members
    #         if current_count > 0:
    #             cluster_center = np.mean(pca_features[samples_in_cluster], axis=0)
    #             logger.info(f"Cluster {j:2d}: center={np.round(cluster_center[:3], 2)} | "
    #                     f"size={current_count:2d} | "
    #                     f"overlap={current_count/m:.1f}x | "
    #                     f"{'FULL' if current_count == max_per_cluster else 'OK'}")
    #         else:
    #             logger.info(f"Cluster {j:2d}: EMPTY (no samples assigned)")
        
    #     # Step 4: Compute new centroids based on assignment matrix
    #     new_centroids = []
    #     for j in range(n_clusters):
    #         samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
    #         if len(samples_in_cluster) > 0:
    #             new_centroid = np.mean(pca_features[samples_in_cluster], axis=0)
    #             new_centroids.append(new_centroid)
    #         else:
    #             logger.warning(f"Group {j} has no samples assigned to it!")
        
    #     # Step 5: Find samples closest to new centroids
    #     representative_indices = []
    #     cluster_to_representative = {}  # Track which sample represents each cluster
    #     for j, new_centroid in enumerate(new_centroids):
    #         # Reshape new_centroid to be 2D for euclidean_distances
    #         centroid_reshaped = new_centroid.reshape(1, -1)
    #         # Calculate distances from all samples to this centroid
    #         distances_to_new_centroid = euclidean_distances(pca_features, centroid_reshaped).flatten()
    #         closest_idx = np.argmin(distances_to_new_centroid)
    #         representative_indices.append(closest_idx)
    #         cluster_to_representative[j] = closest_idx  # Store mapping

    #     # NEW LOGGING: Log which sample represents each cluster
    #     logger.info("\nCluster Representatives (which sample represents each cluster):")
    #     for cluster_id, sample_id in sorted(cluster_to_representative.items()):
    #         logger.info(f"Cluster {cluster_id}: Represented by Sample {sample_id}")

    #     # # Log representative distribution across clusters
    #     # representative_distribution = np.sum(assignment_matrix[representative_indices], axis=0)
    #     # logger.info("\nRepresentative distribution across clusters:")
    #     # for j in range(n_clusters):
    #     #     if representative_distribution[j] > 0:
    #     #         logger.info(f"Cluster {j}: {representative_distribution[j]} representative(s)")
        
    #     # Get representative samples
    #     representative_samples = [range_samples[idx] for idx in representative_indices]
        
    #     # Return representatives and assignment matrix
    #     return representative_samples, assignment_matrix.T

    # Test with visual logs (Centroid)
    def perform_group_testing(self, range_samples: list, n_clusters=5, m=3):
        """Improved group testing with balanced overlapping clusters"""
        # Step 1: PCA 
        flat_features = [sample.numpy().flatten() if isinstance(sample, torch.Tensor) 
                        else np.array(sample).flatten() 
                        for sample in range_samples]

        # NEW: Log sample features before PCA
        logger.info("\nOriginal samples:")
        for i, features in enumerate(flat_features):
            # Show full feature vector
            logger.info(f"Sample {i}: {np.round(features, 2)}")
        
        n_components = min(len(range_samples)-1, self.configs.pca_components)
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(flat_features)

        # NEW: Log PCA features
        logger.info("\nSamples after PCA:")
        for i, features in enumerate(pca_features):
            # Show full PCA feature vector
            logger.info(f"Sample {i}: {np.round(features, 2)}")
        
        logger.info(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")
        
        # Step 2: Run k-means to get cluster centers
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20)
        kmeans.fit(pca_features)
        centroids = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_  # Get cluster assignments from k-means

        # NEW: Visual cluster membership from k-means
        logger.info("\nInitial K-means Clustering:")
        cluster_members = {}
        for i in range(n_clusters):
            cluster_members[i] = np.where(cluster_labels == i)[0]
            logger.info(f"Cluster {i}: Samples {cluster_members[i].tolist()}")

        # NEW: Visual representation of clusters
        logger.info("\nClusters Visualization (samples → clusters):")
        for i in range(len(range_samples)):
            logger.info(f"Sample {i} → Cluster {cluster_labels[i]}")

        # NEW: Log centroids with nearby samples
        logger.info("\nCentroids with nearby samples:")
        for j, centroid in enumerate(centroids):
            # Show full centroid vector
            
            # Find samples closest to this centroid
            centroid_reshaped = centroid.reshape(1, -1)
            distances_to_centroid = euclidean_distances(pca_features, centroid_reshaped).flatten()
            closest_samples = np.argsort(distances_to_centroid)[:3]  # Get 3 closest samples
            
            logger.info(f"Centroid {j}: {np.round(centroid, 2)}  # Near samples {', '.join(map(str, closest_samples))}")
        
        # Step 3: Calculate distances to all centroids for each sample
        distances = euclidean_distances(pca_features, centroids)

        # NEW: Log distance matrix
        logger.info("\nDistances:")
        # Create header
        header = "             "
        for j in range(n_clusters):
            header += f"Centroid {j}  "
        logger.info(header)
        
        # Log each sample's distances
        for i, sample_distances in enumerate(distances):
            row = f"Sample {i}:    "
            for d in sample_distances:
                row += f"{d:.2f}       "
            logger.info(row)
        
        # Step 4: Create balanced assignment matrix
        n_samples = len(range_samples)
        
        # Set maximum samples per cluster (for balance)
        max_per_cluster = int(np.ceil((n_samples * m / n_clusters) * 1.5))
        
        # Initialize assignment matrix
        assignment_matrix = np.zeros((n_samples, n_clusters), dtype=np.uint8)
        
        # Track how many samples are in each cluster
        cluster_counts = np.zeros(n_clusters, dtype=int)
        sample_cluster_counts = np.zeros(n_samples, dtype=int)  # Tracks how many clusters each sample belongs to
        
        # First pass: Assign each sample to its m closest clusters
        logger.info("\nAssigning each sample to its %d closest clusters:", m)
        for i in range(n_samples):
            # Get m closest clusters
            closest_clusters = np.argsort(distances[i])[:m]
            
            # Assign to these clusters
            logger.info(f"Sample {i}: closest clusters {closest_clusters.tolist()} with distances {np.round(distances[i][closest_clusters], 2)}")
            for j in closest_clusters:
                assignment_matrix[i, j] = 1
                cluster_counts[j] += 1
                sample_cluster_counts[i] += 1  # Track per-sample count

        # After balancing, verify all samples have exactly m clusters
        for i in range(n_samples):
            assert np.sum(assignment_matrix[i]) == m, f"Sample {i} has {np.sum(assignment_matrix[i])} clusters (expected {m})"

        # NEW FORMATTED MATRIX LOG - add this new section
        logger.info("\nAssignment Matrix (visually formatted):")
        # Create header
        header = "           "
        for j in range(n_clusters):
            header += f"Cluster {j}  "
        logger.info(header)
        
        # Create each row
        for i in range(n_samples):
            row = f"Sample {i}:     "
            for j in range(n_clusters):
                row += f"{assignment_matrix[i,j]}         "
            logger.info(row)
        
        # NEW LOGGING 1: Assignment Matrix Structure
        logger.info(f"\nAssignment Matrix (Groups x Samples) - m={m} overlapping:")
        logger.info("Assignment Matrix Structure:")
        for j in range(n_clusters):
            row = assignment_matrix[:, j].tolist()  # Get samples for this group
            logger.info(f"Group {j:2d}: {row}")
        
        # NEW LOGGING 2: Sample-to-Group Mappings
        logger.info(f"\nSample-to-Group Assignments (verifying m={m} overlapping):")
        for i in range(n_samples):
            assigned_groups = np.where(assignment_matrix[i, :] == 1)[0].tolist()
            logger.info(f"Sample {i:2d}: assigned to groups {assigned_groups} (count: {len(assigned_groups)})")
        
        
        
        # Original logging 
        logger.info(f"\nFinal clusters after balancing (max per cluster: {max_per_cluster}):")
        for j in range(n_clusters):
            # Get cluster members
            samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
            current_count = len(samples_in_cluster)
            
            # Calculate cluster center based on actual members
            if current_count > 0:
                cluster_center = np.mean(pca_features[samples_in_cluster], axis=0)
                logger.info(f"Cluster {j:2d}: center={np.round(cluster_center[:3], 2)} | "
                        f"size={current_count:2d} | "
                        f"overlap={current_count/m:.1f}x | "
                        f"{'FULL' if current_count == max_per_cluster else 'OK'}")
            else:
                logger.info(f"Cluster {j:2d}: EMPTY (no samples assigned)")
        
        # Step 4: Compute new centroids based on assignment matrix
        new_centroids = []
        logger.info("\nComputing new centroids based on final assignment matrix:")
        for j in range(n_clusters):
            samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
            if len(samples_in_cluster) > 0:
                new_centroid = np.mean(pca_features[samples_in_cluster], axis=0)
                new_centroids.append(new_centroid)
                logger.info(f"New Centroid {j}: {np.round(new_centroid, 2)} (from samples {samples_in_cluster.tolist()})")
            else:
                logger.warning(f"Group {j} has no samples assigned to it!")
        
        # Step 5: Find samples closest to new centroids
        representative_indices = []
        cluster_to_representative = {}  # Track which sample represents each cluster
        
        logger.info("\nSelecting representatives for each cluster:")
        for j, new_centroid in enumerate(new_centroids):
            # Reshape new_centroid to be 2D for euclidean_distances
            centroid_reshaped = new_centroid.reshape(1, -1)
            # Calculate distances from all samples to this centroid
            distances_to_new_centroid = euclidean_distances(pca_features, centroid_reshaped).flatten()
            
            # Get all samples sorted by distance to this centroid
            sorted_indices = np.argsort(distances_to_new_centroid)
            closest_idx = sorted_indices[0]
            
            # Log the top 3 closest samples and their distances
            top_3_indices = sorted_indices[:3]
            top_3_distances = distances_to_new_centroid[top_3_indices]
            
            logger.info(f"Cluster {j} - Closest samples to new centroid:")
            for rank, (idx, dist) in enumerate(zip(top_3_indices, top_3_distances)):
                logger.info(f"  Rank {rank+1}: Sample {idx} (distance: {dist:.4f}){' ← SELECTED' if rank == 0 else ''}")
            
            representative_indices.append(closest_idx)
            cluster_to_representative[j] = closest_idx  # Store mapping

        # NEW LOGGING: Log which sample represents each cluster
        logger.info("\nCluster Representatives (which sample represents each cluster):")
        for cluster_id, sample_id in sorted(cluster_to_representative.items()):
            logger.info(f"Cluster {cluster_id}: Represented by Sample {sample_id}")
            
        # NEW: Log where each representative appears in multiple clusters
        logger.info("\nCross-cluster membership of representatives:")
        for rep_idx in representative_indices:
            member_of_clusters = np.where(assignment_matrix[rep_idx] == 1)[0].tolist()
            logger.info(f"Sample {rep_idx} (representative) is a member of clusters: {member_of_clusters}")

        
        # Get representative samples
        representative_samples = [range_samples[idx] for idx in representative_indices]
        
        # Return representatives and assignment matrix
        return representative_samples, assignment_matrix.T

    def create_transformed_dataset_pkl(self, transformed_samples, selected_indices, true_labels, tmp_dir="./tmp_gt"):
        """Create a dataset with transformed samples after group testing.
        
        Args:
            transformed_samples: List of lists of representative samples
            true_labels: Binary membership labels (1=IN, 0=OUT)
            selected_indices: Original audit indices
        
        Returns:
            Path to saved dataset, metadata path, number of total samples
        """
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Get original class labels from the audit dataset
        original_classes = []
        for idx in selected_indices:
            data_idx = self.audit_data_indices[idx]
            class_label = self.handler.population.y[data_idx].item() 
            original_classes.append(class_label)
        
        # Flatten all samples and maintain their class labels
        all_samples = []
        all_labels = []
        sample_to_range = []
        range_to_samples = {}
        
        # Each range now has a variable number of samples (after group testing)
        for range_idx, (representatives, class_label) in enumerate(zip(transformed_samples, original_classes)):
            start_idx = len(all_samples)
            all_samples.extend(representatives)  # These are the group-tested representatives
            all_labels.extend([class_label] * len(representatives))
            sample_to_range.extend([range_idx] * len(representatives))
            range_to_samples[range_idx] = list(range(start_idx, start_idx + len(representatives)))
        
        # Convert to tensors
        x = torch.stack(all_samples) if all_samples else torch.tensor([])
        y = torch.tensor(all_labels) if all_labels else torch.tensor([])
        
        # Get range masks for classification - still use the original sample's membership
        range_masks = []
        for idx in selected_indices:
            range_masks.append(self.in_indices_masks[idx])
        
        # Create metadata structure for the group-tested samples
        metadata = {
            "original_splits": ["unknown"] * len(all_samples),
            "original_idx": list(range(len(all_samples))),
            "raw_data": None,
            # Additional metadata for RaMIA
            "sample_to_range": sample_to_range,
            "range_to_samples": range_to_samples,
            "range_masks": range_masks,
            "range_membership": {i: label for i, label in enumerate(true_labels)},
            "original_classes": original_classes,
            # Add explicit tracking that this is from group testing
            "group_testing": True,
            "original_indices": selected_indices  # Keep explicit link to original indices
        }
        
        # Create dataset
        transformed_dataset = TransformedDataset(x=x, y=y, metadata=metadata)
        
        # Save dataset
        transformed_dataset_path = os.path.join(tmp_dir, "ramia_transformed_dataset.pkl")
        with open(transformed_dataset_path, "wb") as f:
            pickle.dump(transformed_dataset, f)
        
        metadata_path = os.path.join(tmp_dir, "ramia_metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        return transformed_dataset_path, metadata_path, len(all_samples)

    def _prepare_shadow_models(self:Self) -> None:
        """Prepare shadow models for RMIA attack."""
        logger.info("Preparing shadow models for RaMIA-RMIA attack")

        # Get all available indices for attack dataset
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices=self.online,
                                                                       include_test_indices=self.online)

        # Train shadow models
        logger.info(f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models=self.num_shadow_models,
            shadow_population=self.attack_data_indices,
            training_fraction=self.training_data_fraction,
            online=self.online)
            
        # Load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset using RMIA strategy."""
        # If we already have one run, we don't need to check for shadow models as logits are stored
        if not self.load_for_optuna:
            self._prepare_shadow_models()
            
        # Store original population dataset
        self.original_population = self.handler.population
        self.original_population_size = self.handler.population_size
        
        # Map original audit indices to subset indices
        self.original_to_subset = {
            original_idx: subset_idx
            for subset_idx, original_idx in enumerate(self.original_population.metadata["original_idx"])
        }

        # Validate audit indices
        for idx in self.audit_dataset["data"]:
            if idx not in self.original_to_subset:
                raise ValueError(f"Audit index {idx} not found in population subset")
    
        valid_audit_indices = []
        valid_in_members = []
        valid_out_members = []

        # Track positions in the filtered audit dataset
        for pos, original_idx in enumerate(self.audit_dataset["data"]):
            if original_idx in self.original_to_subset:
                subset_idx = self.original_to_subset[original_idx]
                valid_audit_indices.append(subset_idx)
                # Check membership based on original dataset's labels
                if pos in self.audit_dataset["in_members"]:
                    valid_in_members.append(len(valid_audit_indices)-1)  # Position in filtered list
                if pos in self.audit_dataset["out_members"]:
                    valid_out_members.append(len(valid_audit_indices)-1)  # Position in filtered list
            else:
                logger.warning(f"Audit index {original_idx} not found in population. Skipping.")

        # Update audit data indices and membership
        self.audit_data_indices = np.array(valid_audit_indices)
        self.in_members = np.array(valid_in_members)
        self.out_members = np.array(valid_out_members)

        logger.info("Create masks for all IN and OUT samples")
        self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_data_indices)

        count_in_samples = np.count_nonzero(self.in_indices_masks)
        if count_in_samples > 0:
            logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
            logger.info("Potential contamination in offline attack!")     


        # Prepare RMIA auxiliary data in offline mode
        if self.online is False:
            # compute the ratio of p(z|theta) (target model) to p(z)=sum_{theta'} p(z|theta') (shadow models)
            # for all points in the attack dataset output from signal: # models x # data points x # classes
            if not self.load_for_optuna:
                logits_theta, logits_shadow_models, z_true_labels = self._prepare_offline_aux_attack_logits()
            else:
                logits_theta = np.load(f"{self.attack_folder_path}/logits_theta.npy")
                logits_shadow_models = np.load(f"{self.attack_folder_path}/logits_shadow_models.npy")
                z_true_labels = np.load(f"{self.attack_folder_path}/z_true_labels.npy")

            # collect the softmax output of the correct class
            n_attack_points = len(z_true_labels)
            p_z_given_theta = softmax_logits(logits_theta, self.temperature)[:,np.arange(n_attack_points),z_true_labels]

            # collect the softmax output of the correct class for each shadow model
            sm_logits_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]
            p_z_given_shadow_models = np.array([x[np.arange(n_attack_points),z_true_labels] for x in sm_logits_shadow_models])

            # evaluate the marginal p(z)
            p_z = np.mean(p_z_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_z_given_shadow_models.squeeze()
            p_z = 0.5*((self.offline_a + 1) * p_z + (1-self.offline_a))

            self.ratio_z = p_z_given_theta / (p_z + self.epsilon)


        # Generate transformed ranges
        # num_samples = self.range_num_audit_samples
        num_samples = len(self.audit_data_indices)
        self.transformed_ranges, self.range_labels, self.selected_indices = \
            self.create_transformed_dataset(num_samples)
        
        # Validate results
        if len(self.transformed_ranges) == 0:
            raise RuntimeError("No ranges generated")
        if sum(self.range_labels) == 0:
            raise ValueError("No IN members in ranges")
        
        logger.info(f"Prepared {len(self.transformed_ranges)} ranges "
                    f"({sum(self.range_labels)} IN, {len(self.range_labels)-sum(self.range_labels)} OUT)")
            
        # Create a generic dataset of all transformed samples
        self.transformed_dataset_path, self.metadata_path, self.num_transformed_samples = \
            self.create_transformed_dataset_pkl(self.transformed_ranges, self.selected_indices, self.range_labels) 
        
        # Load the transformed dataset into memory
        with open(self.transformed_dataset_path, "rb") as f:
            self.transformed_dataset = pickle.load(f)

        # Set the transformed dataset as the handler's population
        self.handler.population = self.transformed_dataset
        self.handler.population_size = len(self.transformed_dataset)
        
        # Load metadata for mapping between samples and ranges
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        # Create indices for all transformed samples
        self.transformed_indices = np.arange(self.num_transformed_samples)


    def _prepare_offline_aux_attack_logits(self:Self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the logits for the offline attack on the auxiliary dataset.
        
        This exactly follows the RMIA implementation to maintain consistency.
        """
        # Subsample the attack data based on the fraction
        logger.info(f"Subsampling attack data from {len(self.attack_data_indices)} points")
        n_points = int(self.attack_data_fraction * len(self.attack_data_indices))
        chosen_attack_data_indices = np.random.choice(self.attack_data_indices, n_points, replace=False)
        logger.info(f"Number of attack data points after subsampling: {len(chosen_attack_data_indices)}")

        # Get the true label indices
        z_true_labels = self.handler.get_labels(chosen_attack_data_indices)
        assert np.issubdtype(z_true_labels.dtype, np.integer)
        f_z_true_labels = f"{self.attack_folder_path}/z_true_labels.npy"
        np.save(f_z_true_labels, z_true_labels)

        # Run points through real model to collect the logits
        logits_theta = np.array(self.signal([self.target_model], self.handler, chosen_attack_data_indices))

        # Store the ratio of p(z|theta) to p(z) for the audit dataset to be used in optuna
        f_logits_theta = f"{self.attack_folder_path}/logits_theta.npy"
        np.save(f_logits_theta, logits_theta)

        # Run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, self.handler, chosen_attack_data_indices)
        f_logits_sm = f"{self.attack_folder_path}/logits_shadow_models.npy"
        np.save(f_logits_sm, logits_shadow_models)

        return logits_theta, logits_shadow_models, z_true_labels

    def _prepare_offline_audit_logits(self:Self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the logits for the transformed samples in offline mode."""
        # Run target points through target model to get logits
        logits_theta = np.array(self.signal([self.target_model], self.handler, self.transformed_indices))
        f_logits_theta = f"{self.attack_folder_path}/logits_audit_theta.npy"
        np.save(f_logits_theta, logits_theta)

        # Run points through shadow models and collect the logits
        logits_shadow_models = self.signal(self.shadow_models, self.handler, self.transformed_indices)
        f_logits_sm = f"{self.attack_folder_path}/logits_audit_shadow_models.npy"
        np.save(f_logits_sm, logits_shadow_models)

        return logits_theta, logits_shadow_models

    
    def cleanup(self):
        """Restore original handler state"""
        self.handler.population = self.original_population
        self.handler.population_size = self.original_population_size
        logger.info("Restored original handler state")

    def _offline_attack(self:Self) -> None:
        """Perform the offline attack following RMIA logic exactly."""
        logger.info("Running RMIA offline attack for transformed samples")

        # Get logits for transformed samples
        if not self.load_for_optuna:
            logits_theta, logits_shadow_models = self._prepare_offline_audit_logits()
        else:
            logits_theta = np.load(f"{self.attack_folder_path}/logits_audit_theta.npy")
            logits_shadow_models = np.load(f"{self.attack_folder_path}/logits_audit_shadow_models.npy")

        # Collect the ground truth class labels
        ground_truth_indices = self.handler.get_labels(self.transformed_indices)
        assert np.issubdtype(ground_truth_indices.dtype, np.integer)

        # Get softmax probabilities for target model
        n_samples = len(self.transformed_indices)
        p_x_given_target_model = softmax_logits(logits_theta, self.temperature)[:,np.arange(n_samples),ground_truth_indices]

        # Get softmax probabilities for shadow models
        sm_shadow_models = [softmax_logits(x, self.temperature) for x in logits_shadow_models]
        p_x_given_shadow_models = np.array([x[np.arange(n_samples),ground_truth_indices] for x in sm_shadow_models])

        # Evaluate marginal p_out(x) by averaging shadow model outputs
        p_x_out = np.mean(p_x_given_shadow_models, axis=0) if len(self.shadow_models) > 1 else p_x_given_shadow_models.squeeze()

        # Compute p(x) using offline_a parameter
        p_x = 0.5*((self.offline_a + 1) * p_x_out + (1-self.offline_a))

        # Compute ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_target_model / (p_x + self.epsilon)

        # For each x, compare with ratio of all z points
        likelihoods = ratio_x.T / self.ratio_z

        # Compute membership scores using gamma threshold
        self.all_sample_scores = np.mean(likelihoods > self.gamma, axis=1)


    def run_attack(self:Self) -> MIAResult:
        """
        Runs the Range Membership Inference Attack with improved scoring.

        Returns
        -------
        Result object containing the metric results, including predictions,
        true labels, and signal values.
        """
        # Create transformed masks for tracking membership
        self.transformed_masks = np.array([self.metadata["range_masks"][range_idx] for range_idx in self.metadata["sample_to_range"]])

        # Verify dimensions match
        assert len(self.transformed_masks) == self.num_transformed_samples, \
            "transformed_masks length mismatch with transformed samples"
        
        # Run RMIA offline attack to get all sample scores
        self._offline_attack()

        # Calculate range scores by applying QGT decoding to each range
        range_scores = []
        for range_idx, sample_indices in tqdm(self.metadata["range_to_samples"].items(), desc="Processing ranges"):
            # Get scores for all samples in this range
            range_sample_scores = self.all_sample_scores[sample_indices]
            
            # Get original sample's H matrix
            original_idx = self.selected_indices[range_idx]
            H = self.parity_matrices.get(original_idx, None)
            
            # group_scores = self.decoder.decode(H, range_sample_scores)
            if self.decoder == "gt":  
                group_scores = self.gt_decoder.gt_decode(H, range_sample_scores)
            else:
                group_scores = self.qgt_decoder.qgt_decode(H, range_sample_scores)


            logger.info(f"Range: {range_idx}, Scores: {group_scores}")
            range_scores.append(group_scores)

        # Convert to numpy array
        range_scores = np.array(range_scores)

        # Generate thresholds based on range scores
        self.thresholds = np.linspace(np.min(range_scores), np.max(range_scores), 1000)

        # Create prediction matrix (num_thresholds x num_ranges)
        predictions = (range_scores.reshape(-1, 1) > self.thresholds.reshape(1, -1)).T

        # Prepare true labels (original range membership)
        true_labels = np.array(self.range_labels, dtype=np.int32)

        # Get original audit indices for result tracking
        audit_indices = np.array(self.selected_indices)

        # Cleanup and restore original handler state
        self.cleanup()

        # Create final attack result
        return MIAResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=range_scores,
            audit_indices=audit_indices,
            metadata={
                "attack_type": "RaMIA_GT",
                "num_ranges": len(range_scores),
                "num_transforms": self.num_transforms,
                "rmia_gamma": self.gamma,
                "rmia_offline_a": self.offline_a
            }
        )
        
    def reset_attack(self: Self, config:BaseModel) -> None:
        """Reset attack to initial state."""
        # Assign the new configuration parameters to the object
        for key, value in config.model_dump().items():
            setattr(self, key, value)

        # New hyperparameters have been set, prepare the attack again
        self.prepare_attack()