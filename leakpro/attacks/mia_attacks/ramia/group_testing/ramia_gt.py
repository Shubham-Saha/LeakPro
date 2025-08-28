"""Implementation of the RaMIA attack using RMIA as the base membership inference method."""

from typing import Literal, Callable, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm

import optuna
import os
import torch
import pickle

from leakpro.schemas import OptunaConfig, avg_tpr_at_low_fpr
from leakpro.input_handler.modality_extensions.image_extension import ImageExtension
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
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
from sklearn.metrics.pairwise import manhattan_distances
import torchvision.models as models
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve


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
        range_num_audit_samples: int = Field(default=50, ge=5, description="Number of audit samples to use for range membership inference") # Range transformation parameters
        dataset_type: Literal["image", "tabular", "text"] = Field(default="image", description="Type of dataset being analyzed") # Dataset type configuration
        transform_type: Literal["standard", "cifar", "custom"] = Field(default="cifar", description="Type of transformation to apply to images") # Image transformation configuration
        optuna_config: OptunaConfig = Field(default=OptunaConfig()) # Optuna configuration for hyperparameter search
        objective: Optional[Callable[[MIAResult], float]] = Field(default=None, description="Objective function for optimization")
        augmentation: Literal["relaxed", "aggressive"] = Field(default="relaxed", description="The type of augmentation pool to use.")

        # Trimming parameters
        trim_lower_percentile: int = Field(default=20, ge=0, le=100, description="Lower percentile for trimming (for synthetic data)")
        trim_auto_tune: bool = Field(default=False, description="Whether to automatically tune trimming parameters")
        
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

        #select decoder
        decoder: Literal["gt", "qgt"] = Field(default="gt", description="Select the group testing decoder to use for RaMIA.")

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
            # self.range_transforms = self.image_extension.get_transform(self.transform_type, self.augmentation, **transform_config)
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
        # Calculate natural class distribution
        total_audit_samples = len(self.audit_data_indices)
        in_ratio = len(self.in_members) / total_audit_samples

        # Calculate sample sizes based on natural distribution
        num_in = int(num_samples * in_ratio)
        num_out = num_samples - num_in

        # Ensure we don't exceed available samples
        num_in = min(num_in, len(self.in_members))
        num_out = min(num_out, len(self.out_members))

        # Adjust if total samples < requested
        if (num_in + num_out) < num_samples:
            shortfall = num_samples - (num_in + num_out)
            # Distribute remaining samples proportionally
            add_in = min(shortfall, len(self.in_members) - num_in)
            add_out = min(shortfall - add_in, len(self.out_members) - num_out)
            num_in += add_in
            num_out += add_out
            logger.warning(f"Adjusted sample sizes to {num_in} IN and {num_out} OUT")


        # Randomly select indices from IN and OUT members
        self.selected_in = np.random.choice(self.in_members, num_in, replace=False)
        self.selected_out = np.random.choice(self.out_members, num_out, replace=False)
        self.selected_indices = np.concatenate([self.selected_in, self.selected_out])
        # self.selected_indices = np.concatenate([self.in_members, self.out_members])

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
                range_samples_pretrain = []
                for transform_idx in range(self.num_transforms):
                    # Apply transformations to the image
                    transformed, transformed_pretrain = self.range_transforms(sample, transform_idx) # New Design
                    # transformed, _ = self.range_transforms(sample, transform_idx) # Old Design
                    range_samples.append(transformed)
                    range_samples_pretrain.append(transformed_pretrain)

                logger.info(f"\nSample {idx}: Created {len(range_samples)} transformed samples")
                ##################### Pre-train model design ##########################
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                feature_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                modules = list(feature_extractor.children())[:-1] # Remove the final classification layer
                feature_extractor = torch.nn.Sequential(*modules)
                feature_extractor.eval() # Set to evaluation mode
                feature_extractor.to(device)

                all_feature_vectors = []
                batch_size_for_fe = 32  # Process in batches for feature extraction efficiency
                batch_for_fe_input_tensors = [] # This will hold tensors ready for the feature_extractor

                for i in range(self.num_transforms): 
                    batch_for_fe_input_tensors.append(range_samples_pretrain[i])

                    # If batch is full or it's the last sample, process the batch
                    if len(batch_for_fe_input_tensors) == batch_size_for_fe or i == self.num_transforms - 1:
                        if batch_for_fe_input_tensors: # Ensure batch is not empty
                            # Stack tensors to create a batch and move to device
                            batch_tensor_for_fe = torch.stack(batch_for_fe_input_tensors).to(device)
                            
                            with torch.no_grad(): # Perform feature extraction
                                features_output = feature_extractor(batch_tensor_for_fe)
                            
                            # Post-process features (flatten if necessary)
                            if features_output.ndim > 2: # e.g. (batch_size, num_features, 1, 1)
                                features_flat = features_output.view(features_output.size(0), -1)
                            else: # e.g. (batch_size, num_features)
                                features_flat = features_output
                            
                            all_feature_vectors.extend(features_flat.cpu().numpy()) # Store features as NumPy arrays
                            batch_for_fe_input_tensors = [] # Reset batch

                all_feature_vectors_np = np.array(all_feature_vectors)
                ##################### Pre-train model design ##########################

                # Perform group testing for those transformed samples
                representatives, H = self.perform_group_testing(range_samples, all_feature_vectors_np) # New Design
                # representatives, H = self.perform_group_testing(range_samples) # Old Design
                

                # Store parity matrix and representatives
                self.parity_matrices[idx] = H  # Now stores group info for representatives     
                transformed_samples.append(representatives)
                # transformed_samples.append(range_samples)
                true_labels.append(1 if idx in self.in_members else 0)
                # logger.info(f"Successfully created transforms for index {idx}")
                # logger.info(f"\nCreated {len(representatives)} representatives for sample {idx}")
            except Exception as e:
                logger.warning(f"Error transforming sample {idx}: {e}")
                continue

        logger.info(f"Created {len(transformed_samples)} ranges "
                    f"({sum(true_labels)} IN, {len(true_labels)-sum(true_labels)} OUT)")
        
        return transformed_samples, true_labels, self.selected_indices

    # Medoid approach but clustering strategy is balanced 
    def perform_group_testing(self, range_samples: list, features: np.ndarray, n_clusters=12, m=4): # New Design
    # def perform_group_testing(self, range_samples: list, n_clusters=8, m=3): # Old Design
        """Improved group testing with balanced overlapping clusters"""
        # # Step 1: Flatten features if needed
        # flat_features = [sample.numpy().flatten() if isinstance(sample, torch.Tensor) 
        #                 else np.array(sample).flatten() 
        #                 for sample in range_samples]
        
        # # Step 2: PCA
        # n_components = min(len(range_samples)-1, self.configs.pca_components)
        # pca = PCA(n_components=n_components)
        # # features = pca.fit_transform(flat_features)
        # features = pca.fit_transform(features)

        # Step 2: t-SNE
        n_samples_for_tsne = features.shape[0]
        perplexity_value = min(30.0, float(max(5.0, n_samples_for_tsne - 1))) # Adjust perplexity if too few samples
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=300, learning_rate='auto' if n_samples_for_tsne > 10 else 200.0)
        features = tsne.fit_transform(features)
        # features = tsne.fit_transform(flat_features)

        # Step 3: Run k-means to get cluster centers
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=1234)
        kmeans.fit(features)
        centroids = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        # Print cluster assignments
        logger.info(f"\nCluster assignments:")
        for cluster_id in range(n_clusters):
            members = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            logger.info(f"Cluster {cluster_id}: Samples {members} ({len(members)} samples)")
        
        # Step 4: Calculate distances to all centroids for each sample
        distance = euclidean_distances
        # distance = manhattan_distances
        distances = distance(features, centroids)
        
        # Step 4: Create assignment matrix
        n_samples = len(range_samples)       
        assignment_matrix = np.zeros((n_samples, n_clusters), dtype=np.uint8) # Initialize assignment matrix
        cluster_counts = np.zeros(n_clusters, dtype=int) # Track how many samples are in each cluster
        sample_cluster_counts = np.zeros(n_samples, dtype=int)  # Tracks how many clusters each sample belongs to
        
        # Assign each sample to its m closest clusters
        for i in range(n_samples):
            closest_clusters = np.argsort(distances[i])[:m] # Get m closest clusters
            
            # Assign to these clusters
            for j in closest_clusters:
                assignment_matrix[i, j] = 1
                cluster_counts[j] += 1
                sample_cluster_counts[i] += 1  # Track per-sample count
        
        logger.info(f"\nAssignment Matrix (Groups({n_clusters}) x Samples({n_samples})) - m={m} overlapping:")
        for j in range(n_clusters):
            row = assignment_matrix[:, j].tolist()  # Get samples for this group
            logger.info(f"Group {j:2d}: {row}")
            
        
        for j in range(n_clusters):
            samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
            current_count = len(samples_in_cluster)
            logger.info(f"\nGroup {j:2d}: Samples {samples_in_cluster.tolist()} | size={current_count}")

        # # Sample-to-Group Mappings
        # logger.info(f"\nSample-to-Group Assignments (verifying m={m} overlapping):")
        # for i in range(n_samples):
        #     assigned_groups = np.where(assignment_matrix[i, :] == 1)[0].tolist()
        #     logger.info(f"Sample {i:2d}: assigned to groups {assigned_groups} (count: {len(assigned_groups)})")
        
        
        # Step 5: Find representatives using medoids
        representative_indices = []
        cluster_to_representative = {}  # Track which sample represents each cluster
        for j in range(n_clusters):
            samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
            
            if len(samples_in_cluster) > 0:
                if len(samples_in_cluster) > 1:
                    # Find medoid
                    cluster_features = features[samples_in_cluster]
                    # within_distances = distance(cluster_features)
                    within_distances = distance(cluster_features)
                    sum_distances = np.sum(within_distances, axis=1)
                    medoid_idx = np.argmin(sum_distances)
                    representative_idx = samples_in_cluster[medoid_idx]
                else:
                    representative_idx = samples_in_cluster[0]
                    
                representative_indices.append(representative_idx)
                cluster_to_representative[j] = representative_idx  # Store mapping

        # # Step 5: Compute new centroids based on assignment matrix
        # new_centroids = []
        # for j in range(n_clusters):
        #     samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
        #     if len(samples_in_cluster) > 0:
        #         new_centroid = np.mean(features[samples_in_cluster], axis=0)
        #         new_centroids.append(new_centroid)
        #     else:
        #         logger.warning(f"Group {j} has no samples assigned to it!")

        # # Find samples closest to new centroids
        # representative_indices = []
        # cluster_to_representative = {}

        # for j, new_centroid in enumerate(new_centroids):
        #     # Get samples that actually belong to this cluster
        #     samples_in_cluster = np.where(assignment_matrix[:, j] == 1)[0]
            
        #     if len(samples_in_cluster) > 0:
        #         # Get features only for samples in this cluster
        #         cluster_features = features[samples_in_cluster]
                
        #         # Calculate distances from cluster samples to the centroid
        #         centroid_reshaped = new_centroid.reshape(1, -1)
        #         distances_to_centroid = distance(cluster_features, centroid_reshaped).flatten()
                
        #         # Find the closest sample within this cluster
        #         closest_idx_in_cluster = np.argmin(distances_to_centroid)
        #         representative_idx = samples_in_cluster[closest_idx_in_cluster]
                
        #         representative_indices.append(representative_idx)
        #         cluster_to_representative[j] = representative_idx
        #     else:
        #         logger.warning(f"Cluster {j} is empty - no representative selected!")

        # Log which sample represents each cluster
        logger.info("\nCluster Representatives:")
        for cluster_id, sample_id in sorted(cluster_to_representative.items()):
            logger.info(f"Cluster {cluster_id}: Represented by Sample {sample_id}")
        
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
        num_samples = self.range_num_audit_samples
        # num_samples = len(self.audit_data_indices)
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
        
        # Store in-member and out-member signals
        # (We'll handle these in a RaMIA-specific way later)


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

            fixed_global_threshold = 0.75  # Or 0.75, or 0.85. This is now a hyperparameter.
            binary_outcomes = (range_sample_scores > fixed_global_threshold).astype(np.uint8) # previous >
            logger.info(f"Range {range_idx} | Fixed global threshold: {fixed_global_threshold:.4f} | Binary outcomes: {binary_outcomes} | Range sample scores: {range_sample_scores}")

            # Use the optimal threshold with the decoder
            if self.configs.decoder == "gt":  
                group_scores = self.gt_decoder.gt_decode(H, binary_outcomes)
            else:
                group_scores = self.qgt_decoder.qgt_decode(H, binary_outcomes)


            logger.info(f"Range: {range_idx}, Scores: {group_scores}")
            range_scores.append(group_scores)


        # Convert to numpy array
        range_scores = np.array(range_scores)

        min_value = np.min(range_scores) - 1e-4

        # Generate thresholds based on range scores
        self.thresholds = np.linspace(min_value, np.max(range_scores), 1000)

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