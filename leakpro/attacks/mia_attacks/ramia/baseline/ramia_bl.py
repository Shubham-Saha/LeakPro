"""Implementation of the RaMIA attack using RMIA as the base membership inference method."""

from typing import Literal, Callable, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import norm
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import shutil
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
    
class AttackRaMIA_BL(AbstractMIA):
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
            
            # # Skip if conversion failed
            # if sample is None:
            #     logger.warning(f"Failed to convert sample to PIL image for index {data_idx}")
            #     continue

            # Generate transformed samples
            try:
                range_samples = []
                for transform_idx in range(self.num_transforms):
                    # Apply transformations to the image
                    transformed, _ = self.range_transforms(sample, transform_idx)
                    range_samples.append(transformed)

                transformed_samples.append(range_samples)
                true_labels.append(1 if idx in self.in_members else 0)
                logger.info(f"Successfully created transforms for index {idx}")
            except Exception as e:
                logger.warning(f"Error transforming sample {idx}: {e}")
                continue

        logger.info(f"Created {len(transformed_samples)} ranges "
                    f"({sum(true_labels)} IN, {len(true_labels)-sum(true_labels)} OUT)")
        
        return transformed_samples, true_labels, self.selected_indices


    def create_transformed_dataset_pkl(self, transformed_samples, selected_indices, true_labels, tmp_dir="./tmp_rmia"):
        """Create a dataset with transformed samples that follows CifarDataset structure."""
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Get original class labels from the audit dataset
        original_classes = []
        for idx in selected_indices:
            data_idx = self.audit_data_indices[idx]
            # Get class label directly - simplified
            class_label = self.handler.population.y[data_idx].item() 
            original_classes.append(class_label)
        
        # Flatten all samples and maintain their class labels
        all_samples = []
        all_labels = []
        sample_to_range = []
        range_to_samples = {}
        
        for range_idx, (range_samples, class_label) in enumerate(zip(transformed_samples, original_classes)):
            start_idx = len(all_samples)
            all_samples.extend(range_samples)
            all_labels.extend([class_label] * len(range_samples))
            sample_to_range.extend([range_idx] * len(range_samples))
            range_to_samples[range_idx] = list(range(start_idx, start_idx + len(range_samples)))
        
        # Convert to tensors
        x = torch.stack(all_samples) if all_samples else torch.tensor([])
        y = torch.tensor(all_labels) if all_labels else torch.tensor([])
        
        # Get range masks for classification
        range_masks = []
        for idx in selected_indices:
            range_masks.append(self.in_indices_masks[idx])
        
        # Create metadata structure similar to CifarDataset
        metadata = {
            "original_splits": ["unknown"] * len(all_samples),
            "original_idx": list(range(len(all_samples))),
            "raw_data": None,
            # Additional metadata for RaMIA
            "sample_to_range": sample_to_range,
            "range_to_samples": range_to_samples,
            "range_masks": range_masks,
            "range_membership": {i: label for i, label in enumerate(true_labels)},
            "original_classes": original_classes
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

    def compute_trimmed_average(self: Self, ta_scores: np.ndarray) -> float:
        """Trim scores according to RaMIA paper's recommendations for synthetic data.
        
        For synthetic data, we trim the lowest scoring samples to reduce noise.
        
        Args:
            ta_scores: Array of membership scores for samples in a range
            
        Returns:
            float: Trimmed average score
        """
        if len(ta_scores) == 0:
            return 0.0
        
        # Sort scores in ascending order
        sorted_scores = np.sort(ta_scores)
        
        # Calculate quantile indices
        lower_idx = int(len(sorted_scores) * self.trim_lower_percentile // 100)
        upper_idx = int(len(sorted_scores) * 100 // 100)
        # upper_idx = len(sorted_scores)  # Keep all upper scores
        
        # Trim bottom quantiles (ID)
        trimmed_scores = sorted_scores[lower_idx:upper_idx]
        # trimmed_scores = sorted_scores[:lower_idx]
        
        # Return average of remaining scores
        return np.mean(trimmed_scores) if trimmed_scores.size > 0 else 0.0

    def tune_trimming_parameter(self: Self) -> int:
        """Find the optimal trimming parameter for synthetic data using Optuna."""
        if not self.trim_auto_tune:
            logger.info(f"Using fixed trim lower percentile: {self.trim_lower_percentile}")
            return self.trim_lower_percentile
            
        logger.info("Tuning trimming parameter using Optuna")
        
        def objective(trial):
            # Sample a value for the trimming percentile
            percentile = trial.suggest_int("trim_lower_percentile", 5, 40)
            
            # Compute range scores with this percentile
            range_scores = []
            for range_idx, sample_indices in self.metadata["range_to_samples"].items():
                range_sample_scores = self.all_sample_scores[sample_indices]
                
                # Sort scores in ascending order
                sorted_scores = np.sort(range_sample_scores)
                
                # Calculate lower index to keep
                lower_idx = int(len(sorted_scores) * percentile // 100)
                # upper_idx = len(sorted_scores)  # Keep all upper scores
                # upper_idx = int(len(sorted_scores) * 100 // 100)
                
                # Trim bottom quantiles
                trimmed_scores = sorted_scores[:lower_idx]
                # trimmed_scores = sorted_scores[lower_idx:upper_idx]
                
                # Calculate average of remaining scores
                score = np.mean(trimmed_scores) if trimmed_scores.size > 0 else 0.0
                range_scores.append(score)
            
            # Convert to numpy array
            range_scores = np.array(range_scores)
            
            # Generate thresholds
            min_score = np.min(range_scores)
            max_score = np.max(range_scores)
            thresholds = np.linspace(min_score, max_score, 1000)
            
            # Prepare binary predictions matrix
            predictions = (range_scores.reshape(-1, 1) > thresholds.reshape(1, -1)).T
            
            # Create temporary MIAResult for evaluation
            result = MIAResult(
                predicted_labels=predictions,
                true_labels=np.array(self.range_labels, dtype=np.int32),
                predictions_proba=None,
                signal_values=range_scores,
                audit_indices=np.array(self.selected_indices)
            )
            
            # Return the objective value
            return self.objective(result)
        
        # Create and run the Optuna study
        study = optuna.create_study(
            direction=self.configs.optuna_config.direction, 
            pruner=self.configs.optuna_config.pruner,
            sampler=optuna.samplers.TPESampler(seed=self.configs.optuna_config.seed)
        )
        
        study.optimize(
            objective,
            n_trials=self.configs.optuna_config.n_trials
        )
        
        # Get the best trim percentile
        best_percentile = study.best_params["trim_lower_percentile"]
        logger.info(f"Best trimming percentile: {best_percentile}")
        return best_percentile

    def cleanup(self):
        """Restore original handler state"""
        self.handler.population = self.original_population
        self.handler.population_size = self.original_population_size

        # Delete the temporary directory
        try:
            tmp_dir = "./tmp_rmia"
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
                logger.info(f"Successfully deleted temporary directory: {tmp_dir}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary directory: {e}")
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
        """Runs the Range Membership Inference Attack with RMIA scoring.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and signal values.
        """
        # Expand range_masks to transformed_samples:
        self.transformed_masks = np.array([self.metadata["range_masks"][range_idx] for range_idx in self.metadata["sample_to_range"]])

        # After creating transformed_masks
        assert len(self.transformed_masks) == self.num_transformed_samples, \
            "transformed_masks length mismatch with transformed samples"
        
        # Run RMIA offline attack to get all sample scores
        self._offline_attack()

        # Tune trimming parameters if enabled
        if self.trim_auto_tune:
            self.trim_lower_percentile = self.tune_trimming_parameter()

        # Calculate range scores by applying trimmed average to each range
        range_scores = []
        for range_idx, sample_indices in self.metadata["range_to_samples"].items():
            range_sample_scores = self.all_sample_scores[sample_indices]
            
            # Apply trimmed averaging for synthetic data
            trimmed_score = self.compute_trimmed_average(range_sample_scores)
            range_scores.append(trimmed_score)
            logger.info(f"Range {range_idx}: Trimmed average score: {trimmed_score}")
        
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

        # Ensure we use the stored quantities in future runs
        self.load_for_optuna = True

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
                "attack_type": "RaMIA_RMIA",
                "trim_lower_percentile": self.trim_lower_percentile,
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