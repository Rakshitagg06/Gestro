#!/usr/bin/env python3
"""
Image Augmentation Script for Gesture Recognition Dataset

This script augments images in the gestures folder to create a more robust dataset
for gesture recognition model training. It applies various augmentation techniques
while preserving the original images.

Features:
- Rotation augmentation
- Horizontal flipping
- Brightness/contrast adjustment
- Gaussian noise addition
- Zoom/scale variations
- Slight translation
- Gaussian blur
- Color jittering
- Shadow effects
- Perspective transformation

Author: GitHub Copilot
Date: November 17, 2025
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
import argparse
from typing import List, Tuple
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GestureImageAugmenter:
    """
    A comprehensive image augmentation class for gesture recognition datasets.
    """
    
    def __init__(self, input_dir: str, augmentations_per_image: int = 5, seed: int = 42):
        """
        Initialize the augmenter.
        
        Args:
            input_dir: Path to the gestures directory
            augmentations_per_image: Number of augmented versions to create per original image
            seed: Random seed for reproducibility
        """
        self.input_dir = Path(input_dir)
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Ensure input directory exists
        if not self.input_dir.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
    
    def rotate_image(self, image: np.ndarray, angle_range: Tuple[int, int] = (-15, 15)) -> np.ndarray:
        """
        Rotate image by a small random angle within the specified range.
        """
        angle = random.uniform(angle_range[0], angle_range[1])
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                borderMode=cv2.BORDER_REFLECT_101)
        return rotated
    
    def rotate_slight(self, image: np.ndarray, angle_range: Tuple[int, int] = (-8, 8)) -> np.ndarray:
        """
        Apply very slight rotation for subtle variation.
        """
        angle = random.uniform(angle_range[0], angle_range[1])
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                borderMode=cv2.BORDER_REFLECT_101)
        return rotated
    
    def adjust_brightness_contrast(self, image: np.ndarray, 
                                 brightness_range: Tuple[float, float] = (-30, 30),
                                 contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Adjust brightness and contrast of the image.
        """
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
    
    def add_gaussian_noise(self, image: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to the image.
        """
        noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def zoom_image(self, image: np.ndarray, zoom_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Apply zoom/scale transformation to the image.
        """
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        if zoom_factor > 1.0:
            # Crop to original size (zoom in)
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            cropped = resized[start_y:start_y + height, start_x:start_x + width]
            return cropped
        else:
            # Pad to original size (zoom out)
            pad_x = (width - new_width) // 2
            pad_y = (height - new_height) // 2
            padded = cv2.copyMakeBorder(resized, pad_y, height - new_height - pad_y,
                                      pad_x, width - new_width - pad_x,
                                      cv2.BORDER_REFLECT_101)
            return padded
    
    def translate_image(self, image: np.ndarray, 
                       translation_range: Tuple[int, int] = (-20, 20)) -> np.ndarray:
        """
        Apply random translation to the image.
        """
        tx = random.randint(translation_range[0], translation_range[1])
        ty = random.randint(translation_range[0], translation_range[1])
        
        height, width = image.shape[:2]
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        translated = cv2.warpAffine(image, translation_matrix, (width, height),
                                  borderMode=cv2.BORDER_REFLECT_101)
        return translated
    
    def apply_gaussian_blur(self, image: np.ndarray, 
                           kernel_range: Tuple[int, int] = (3, 7)) -> np.ndarray:
        """
        Apply Gaussian blur to the image.
        """
        kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))  # Odd numbers only
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def color_jitter(self, image: np.ndarray, 
                    hue_shift: int = 10, saturation_factor: float = 0.2) -> np.ndarray:
        """
        Apply color jittering (hue and saturation changes).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust hue
        hue_shift_val = random.randint(-hue_shift, hue_shift)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift_val) % 180
        
        # Adjust saturation
        saturation_factor_val = random.uniform(1 - saturation_factor, 1 + saturation_factor)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor_val, 0, 255)
        
        # Convert back to BGR
        jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return jittered
    
    def add_shadow(self, image: np.ndarray) -> np.ndarray:
        """
        Add random shadow effects to the image.
        """
        height, width = image.shape[:2]
        
        # Create random shadow polygon
        vertices = []
        for _ in range(4):
            x = random.randint(0, width)
            y = random.randint(0, height)
            vertices.append([x, y])
        
        vertices = np.array([vertices], dtype=np.int32)
        
        # Create shadow mask
        shadow_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(shadow_mask, vertices, 255)
        
        # Apply shadow (darken the region)
        shadow_intensity = random.uniform(0.3, 0.7)
        shadow_image = image.copy().astype(np.float32)
        
        for i in range(3):  # For each color channel
            shadow_image[:, :, i] = np.where(shadow_mask == 255,
                                           shadow_image[:, :, i] * shadow_intensity,
                                           shadow_image[:, :, i])
        
        return shadow_image.astype(np.uint8)
    
    def perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply slight perspective transformation.
        """
        height, width = image.shape[:2]
        
        # Define perspective transformation points
        offset = random.randint(5, 15)
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst_points = np.float32([
            [random.randint(0, offset), random.randint(0, offset)],
            [width - random.randint(0, offset), random.randint(0, offset)],
            [width - random.randint(0, offset), height - random.randint(0, offset)],
            [random.randint(0, offset), height - random.randint(0, offset)]
        ])
        
        # Get perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        transformed = cv2.warpPerspective(image, perspective_matrix, (width, height),
                                        borderMode=cv2.BORDER_REFLECT_101)
        return transformed
    
    def augment_single_image(self, image: np.ndarray, augmentation_type: str) -> np.ndarray:
        """
        Apply a single augmentation technique to an image.
        """
        augmentation_functions = {
            'rotation': self.rotate_image,
            'rotation_slight': self.rotate_slight,
            'brightness_contrast': self.adjust_brightness_contrast,
            'gaussian_noise': self.add_gaussian_noise,
            'zoom': self.zoom_image,
            'translation': self.translate_image,
            'gaussian_blur': self.apply_gaussian_blur,
            'color_jitter': self.color_jitter,
            'shadow': self.add_shadow,
            'perspective': self.perspective_transform
        }
        
        if augmentation_type in augmentation_functions:
            return augmentation_functions[augmentation_type](image)
        else:
            logger.warning(f"Unknown augmentation type: {augmentation_type}")
            return image
    
    def get_augmentation_combinations(self) -> List[List[str]]:
        """
        Define combinations of augmentations to apply (no horizontal flipping).
        """
        single_augmentations = [
            ['rotation'],
            ['rotation_slight'],
            ['brightness_contrast'],
            ['gaussian_noise'],
            ['zoom'],
            ['translation'],
            ['gaussian_blur'],
            ['color_jitter'],
            ['shadow'],
            ['perspective']
        ]
        
        combination_augmentations = [
            ['rotation', 'brightness_contrast'],
            ['rotation_slight', 'gaussian_noise'],
            ['zoom', 'color_jitter'],
            ['translation', 'gaussian_blur'],
            ['rotation', 'shadow'],
            ['perspective', 'brightness_contrast'],
            ['rotation_slight', 'zoom'],
            ['gaussian_noise', 'translation'],
            ['color_jitter', 'gaussian_blur'],
            ['shadow', 'perspective']
        ]
        
        return single_augmentations + combination_augmentations
    
    def generate_augmented_filename(self, original_filename: str, aug_index: int, 
                                  aug_types: List[str]) -> str:
        """
        Generate filename for augmented image.
        """
        name, ext = os.path.splitext(original_filename)
        aug_suffix = "_".join(aug_types)
        return f"{name}_aug_{aug_index}_{aug_suffix}{ext}"
    
    def augment_image(self, image_path: Path, output_dir: Path) -> None:
        """
        Augment a single image and save the results.
        """
        try:
            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return
            
            # Get augmentation combinations
            augmentation_combinations = self.get_augmentation_combinations()
            
            # Randomly select combinations for this image
            selected_combinations = random.sample(
                augmentation_combinations, 
                min(self.augmentations_per_image, len(augmentation_combinations))
            )
            
            # Apply augmentations
            for i, aug_types in enumerate(selected_combinations):
                augmented_image = image.copy()
                
                # Apply each augmentation in the combination
                for aug_type in aug_types:
                    augmented_image = self.augment_single_image(augmented_image, aug_type)
                
                # Generate output filename
                output_filename = self.generate_augmented_filename(
                    image_path.name, i + 1, aug_types
                )
                output_path = output_dir / output_filename
                
                # Save augmented image
                cv2.imwrite(str(output_path), augmented_image)
                
        except Exception as e:
            logger.error(f"Error augmenting image {image_path}: {str(e)}")
    
    def augment_dataset(self) -> None:
        """
        Augment all images in the gestures dataset.
        """
        logger.info(f"Starting image augmentation for dataset in: {self.input_dir}")
        
        # Get all gesture class directories
        gesture_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        total_images = 0
        total_augmented = 0
        
        for gesture_dir in gesture_dirs:
            logger.info(f"Processing gesture class: {gesture_dir.name}")
            
            # Get all image files in this gesture directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [
                f for f in gesture_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            logger.info(f"Found {len(image_files)} images in {gesture_dir.name}")
            
            # Process each image
            for image_file in tqdm(image_files, desc=f"Augmenting {gesture_dir.name}"):
                self.augment_image(image_file, gesture_dir)
                total_images += 1
                total_augmented += self.augmentations_per_image
        
        logger.info(f"Augmentation complete!")
        logger.info(f"Original images: {total_images}")
        logger.info(f"Augmented images created: {total_augmented}")
        logger.info(f"Total images in dataset: {total_images + total_augmented}")
    
    def get_dataset_statistics(self) -> dict:
        """
        Get statistics about the dataset before and after augmentation.
        """
        stats = {}
        gesture_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        for gesture_dir in gesture_dirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            all_files = [
                f for f in gesture_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            original_files = [f for f in all_files if '_aug_' not in f.name]
            augmented_files = [f for f in all_files if '_aug_' in f.name]
            
            stats[gesture_dir.name] = {
                'original': len(original_files),
                'augmented': len(augmented_files),
                'total': len(all_files)
            }
        
        return stats


def main():
    """
    Main function to run the image augmentation script.
    """
    parser = argparse.ArgumentParser(description='Augment gesture recognition images')
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./gestures',
        help='Path to the gestures directory (default: ./gestures)'
    )
    parser.add_argument(
        '--augmentations_per_image', 
        type=int, 
        default=5,
        help='Number of augmented versions to create per original image (default: 5)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--stats_only', 
        action='store_true',
        help='Only show dataset statistics without performing augmentation'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize augmenter
        augmenter = GestureImageAugmenter(
            input_dir=args.input_dir,
            augmentations_per_image=args.augmentations_per_image,
            seed=args.seed
        )
        
        if args.stats_only:
            # Show statistics only
            stats = augmenter.get_dataset_statistics()
            print("\nDataset Statistics:")
            print("-" * 50)
            for gesture_class, counts in stats.items():
                print(f"{gesture_class:15} | Original: {counts['original']:3d} | "
                      f"Augmented: {counts['augmented']:3d} | Total: {counts['total']:3d}")
        else:
            # Perform augmentation
            augmenter.augment_dataset()
            
            # Show final statistics
            print("\nFinal Dataset Statistics:")
            print("-" * 50)
            stats = augmenter.get_dataset_statistics()
            for gesture_class, counts in stats.items():
                print(f"{gesture_class:15} | Original: {counts['original']:3d} | "
                      f"Augmented: {counts['augmented']:3d} | Total: {counts['total']:3d}")
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())