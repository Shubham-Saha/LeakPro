"""ImageExtension class for handling image data transformation and augmentation."""

import numpy as np
import os
import torch
from torch import Tensor
import torchvision.transforms as transforms
from PIL import Image
import random
from torchvision.datasets import CIFAR10
import itertools

from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.input_handler.modality_extensions.modality_extension import AbstractModalityExtension
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class ImageExtension(AbstractModalityExtension):
    """Class for handling extra functionality for image data."""

    # Assumes that the data is in shape: batchsize, channels, height, width

    def __init__(self:Self, handler:MIAHandler) -> None:

        super().__init__(handler)
        logger.info("Image extension initialized.")

    def augmentation(self:Self, data:Tensor, n_aug:int) -> Tensor:
        """Augment the data by generating additional samples.

        Args:
        ----
            data (Tensor): The input data tensor to augment.
            n_aug (int): The number of augmented samples to generate.

        Returns:
        -------
            Tensor: The augmented data tensor.

        """
        n_aug = n_aug // len(data) #dummy to pass ruff
        return data

    def get_data(self, dataset, index):
        """Retrieve raw sample using metadata from the PKL dataset."""
        try:
            # Get original index from metadata
            original_idx = dataset.metadata["original_idx"][index]
            # Return raw pixel values (0-255) from metadata
            return dataset.metadata["raw_data"][original_idx]
        except Exception as e:
            logger.error(f"Failed to retrieve raw data at index {index}: {e}")
            return None
    
    def to_pil_image(self, raw_sample):
        """Convert raw numpy array (HWC, uint8) to PIL Image."""
        try:
            # Direct conversion (no normalization needed)
            return Image.fromarray(raw_sample)
        except Exception as e:
            logger.error(f"Failed to convert to PIL image: {e}")
            return None
    

    # def get_transform(self, transform_type="cifar", augmentation="relaxed", **kwargs):
    def get_transform(self, transform_type="cifar", **kwargs):
        """Returns transformation function that applies ONE random transform per ID"""
        if transform_type == "cifar":
            # Define all possible individual transformations
            TRANSFORM_POOL = []

            TRANSFORM_POOL.append(transforms.Grayscale(num_output_channels=3))
            TRANSFORM_POOL.append(transforms.RandomInvert(p=1.0))
            TRANSFORM_POOL.append(transforms.RandomPerspective(distortion_scale=0.6, p=1.0))
            TRANSFORM_POOL.append(transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5))
            TRANSFORM_POOL.append(transforms.RandomPosterize(bits=2, p=1.0))
            TRANSFORM_POOL.append(transforms.RandomSolarize(threshold=80, p=1.0))
            TRANSFORM_POOL.append(transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 3.0)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.6, 1.4), shear=20))

            TRANSFORM_POOL.append(transforms.RandomHorizontalFlip(p=1.0))
            TRANSFORM_POOL.append(transforms.Compose([
                transforms.Pad(padding=2, padding_mode='reflect'),
                transforms.RandomCrop(size=32)
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                transforms.Pad(padding=4, padding_mode='reflect'),
                transforms.RandomCrop(size=32)
            ]))
            TRANSFORM_POOL.append(transforms.RandomRotation(degrees=(-5, 5)))
            TRANSFORM_POOL.append(transforms.RandomRotation(degrees=(-10, 10)))
            TRANSFORM_POOL.append(transforms.RandomRotation(degrees=(-15, 15)))
            TRANSFORM_POOL.append(transforms.ColorJitter(brightness=0.2))
            TRANSFORM_POOL.append(transforms.ColorJitter(contrast=0.2))
            TRANSFORM_POOL.append(transforms.ColorJitter(saturation=0.2))
            TRANSFORM_POOL.append(transforms.ColorJitter(hue=0.05))
            TRANSFORM_POOL.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            TRANSFORM_POOL.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
            TRANSFORM_POOL.append(transforms.ColorJitter(brightness=0.3))
            TRANSFORM_POOL.append(transforms.ColorJitter(contrast=0.3))
            TRANSFORM_POOL.append(transforms.ColorJitter(saturation=0.3))
            TRANSFORM_POOL.append(transforms.ColorJitter(hue=0.1))
            TRANSFORM_POOL.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3))
            TRANSFORM_POOL.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
            TRANSFORM_POOL.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)))
            TRANSFORM_POOL.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, translate=(0.07, 0)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, translate=(0, 0.07)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, translate=(0.07, 0.07)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, translate=(0.12, 0.12)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, shear=(-7, 7, 0, 0)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, shear=(0, 0, -7, 7)))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=0, shear=(-7, 7, -7, 7)))

            geometrics_for_combo = [
                TRANSFORM_POOL[0], # Flip
                TRANSFORM_POOL[2], # PadCrop4
                TRANSFORM_POOL[4], # Rot10
                TRANSFORM_POOL[22] # Affine Translate XY slight
            ]
            colors_for_combo = [
                TRANSFORM_POOL[10],# CJ-BCS-Mild
                TRANSFORM_POOL[17],# CJ-BCSH-Mod
                TRANSFORM_POOL[18] # Blur-Mild
            ]

            for geom_t in geometrics_for_combo:
                for color_t in colors_for_combo:
                    TRANSFORM_POOL.append(transforms.Compose([geom_t, color_t]))

            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[0], # Flip
                TRANSFORM_POOL[4]  # Rot10
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[2], # PadCrop4
                TRANSFORM_POOL[3]  # Rot5
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[0],  # Flip
                TRANSFORM_POOL[22] # Affine Translate XY slight
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[3],  # Rot5
                TRANSFORM_POOL[24] # Affine Scale slight
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[2],  # PadCrop4
                TRANSFORM_POOL[17] # CJ-BCSH-Mod
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[0],  # Flip
                TRANSFORM_POOL[10] # CJ-BCS-Mild
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[5],  # Rot15
                TRANSFORM_POOL[18] # Blur-Mild
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[23], # Affine Translate XY Mod
                TRANSFORM_POOL[24]  # Affine Scale slight
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[0],  # Flip
                TRANSFORM_POOL[2],  # PadCrop4
                TRANSFORM_POOL[10] # CJ-BCS-Mild
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[4],  # Rot10
                TRANSFORM_POOL[22], # Affine Translate XY slight
                TRANSFORM_POOL[16] # CJ-BCS-Mod (using index for moderate BCS)
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[2],  # PadCrop4
                TRANSFORM_POOL[3],  # Rot5
                TRANSFORM_POOL[6]   # Brightness Mild
            ]))
            TRANSFORM_POOL.append(transforms.RandomAffine(degrees=(-5,5), translate=(0.05,0.05), scale=(0.95,1.05), shear=(-5,5,-5,5)))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[0], # Flip
                transforms.RandomAffine(degrees=(-5,5), translate=(0.05,0.05), scale=(0.95,1.05), shear=(-5,5,-5,5))
            ]))
            TRANSFORM_POOL.append(transforms.RandomPerspective(distortion_scale=0.1, p=1.0))
            TRANSFORM_POOL.append(transforms.Compose([
                TRANSFORM_POOL[0], # Flip
                transforms.RandomPerspective(distortion_scale=0.1, p=1.0)
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                transforms.Pad(padding=2, padding_mode='reflect'),
                transforms.RandomCrop(size=32),
                transforms.RandomRotation(degrees=(-7,7)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            ]))
            TRANSFORM_POOL.append(transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.Pad(padding=4, padding_mode='reflect'),
                transforms.RandomCrop(size=32),
                transforms.RandomRotation(degrees=(-10,10)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            ]))

            # TRANSFORM_POOL = [
            #     transforms.RandomAutocontrast(p=1.0),
            #     transforms.RandomSolarize(threshold=128, p=1.0),
            #     transforms.RandomEqualize(p=1.0),
            #     transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
            #     transforms.GaussianBlur(kernel_size=5, sigma=(0.2, 1.0)),
            #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.06),
            #     transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.09),
            #     transforms.RandomPosterize(bits=5, p=1.0),
            #     transforms.RandomResizedCrop(32, scale=(0.55, 1.0), ratio=(0.85, 1.15)),
            #     transforms.RandomHorizontalFlip(p=1.0),
            #     transforms.RandomRotation(degrees=(30)),
            #     transforms.RandomVerticalFlip(p=1.0),
            #     transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0),
            # ]


            # Define normalization schemes and pre-PIL transforms for both target and pretrain
            target_normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            pretrain_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # Pre-PIL transforms for pretrain (resizing for pretrained models)
            pretrain_pre_transforms = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
            
            # =================== NEW: Define Preliminary Transformations ===================

            preliminary_transform = transforms.RandomHorizontalFlip(p=1.0)
            # preliminary_transform = transforms.RandomInvert(p=1.0)
            # preliminary_transform = transforms.Compose([
            #     transforms.Grayscale(num_output_channels=3),
            #     transforms.RandomInvert(p=1.0),
            #     transforms.RandomHorizontalFlip(p=1.0),
            # ])

            # =============================================================================

            def cifar_transform(image: Image.Image, transform_idx: int) -> tuple[Tensor, Tensor]:
                """
                Applies a selected transformation and returns both target and pretrain normalized versions.
                Input 'image' must be a PIL Image.
                Returns tuple: (target_tensor, pretrain_tensor)
                """
                # Set seeds for reproducible transforms
                seed = transform_idx
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)

                selected_idx = (transform_idx - 1) % len(TRANSFORM_POOL)
                augmentation_transform = TRANSFORM_POOL[selected_idx]
                # augmentation_transform = np.random.choice(TRANSFORM_POOL)
                # base_augmented_image = preliminary_transform(image)
                # augmented_image = augmentation_transform(base_augmented_image)
                augmented_image = augmentation_transform(image)


                # Step 2: Create target version (32x32, target normalization)
                target_pipeline = transforms.Compose([
                    transforms.ToTensor(),  # PIL -> Tensor [0,1]
                    target_normalization    # Apply target normalization
                ])
                # target_tensor = target_pipeline(augmented_image)
                target_tensor = target_pipeline(augmented_image)

                # Step 3: Create pretrain version (224x224, ImageNet normalization)
                pretrain_pipeline_steps = pretrain_pre_transforms + [
                    transforms.ToTensor(),         # PIL -> Tensor [0,1]
                    pretrain_normalization         # Apply ImageNet normalization
                ]
                pretrain_pipeline = transforms.Compose(pretrain_pipeline_steps)
                pretrain_tensor = pretrain_pipeline(augmented_image)

                return target_tensor, pretrain_tensor
                # return target_tensor

            return cifar_transform
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")

