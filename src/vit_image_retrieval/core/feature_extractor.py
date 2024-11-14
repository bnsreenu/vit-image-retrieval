# feature_extractor.py
# Add these lines at the very beginning of the file
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


import torch
from torch.utils.data import Dataset
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Callable
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    """Dataset for batch processing of images."""
    def __init__(self, image_paths: list, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image_path
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

class ImageFeatureExtractor:
    def __init__(self, device: Optional[str] = None, model_dir: Optional[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            device: Optional device specification ('cuda' or 'cpu')
            model_dir: Optional directory to store the model weights
                      If None, uses a local directory in the user's home folder
        """
        # Set up model directory
        if model_dir is None:
            # Create a directory in user's home directory
            home = Path.home()
            model_dir = home / '.vit_image_retrieval' / 'models'
        else:
            model_dir = Path(model_dir)
            
        try:
            # Create directory with proper permissions
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(model_dir))
            logger.info(f"Using model directory: {model_dir}")
        except Exception as e:
            logger.warning(f"Could not create model directory {model_dir}: {e}")
            logger.warning("Falling back to temporary directory")
            import tempfile
            model_dir = Path(tempfile.gettempdir()) / '.vit_image_retrieval' / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(model_dir))
            logger.info(f"Using temporary model directory: {model_dir}")
        
        # Set up device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize ViT model
        try:
            logger.info("Loading ViT-B/16 model...")
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=weights)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Create a modified forward method to get embeddings instead of classification
        self.original_forward = self.model.forward
        self.model.forward = self._forward_features
        
        self.model.eval()
        self.model.to(self.device)
        
        self.feature_dim = 768  # Fixed for ViT-B/16
        
        # ViT specific transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Initialized ViT feature extractor with dimension: {self.feature_dim}")

    def _forward_features(self, x):
        """Modified forward pass to get embeddings instead of classification."""
        # Process input
        x = self.model._process_input(x)
        n = x.shape[0]

        # Add class token
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.model.encoder(x)

        # Return CLS token embedding
        return x[:, 0]

    @torch.no_grad()
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image."""
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass through model
            features = self.model(image)
            
            # Convert to numpy and normalize
            features = features.cpu().numpy().squeeze()
            
            # Ensure we have the correct feature dimension
            if features.shape != (self.feature_dim,):
                raise ValueError(f"Unexpected feature dimension: {features.shape}")
            
            # L2 normalize the features
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            logger.debug(f"Extracted features shape: {features.shape}")
            logger.debug(f"Features norm: {np.linalg.norm(features)}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
            raise

    def extract_batch_features(self, image_dir: str, progress_callback: Optional[Callable] = None) -> tuple:
        """
        Extract features from all images in a directory with progress reporting.
        
        Args:
            image_dir: Directory containing images
            progress_callback: Optional callback function to report progress
            
        Returns:
            tuple: (features_list, valid_paths)
        """
        # Get all valid image paths
        image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        
        total_images = len(image_paths)
        features_list = []
        valid_paths = []
        
        for idx, img_path in enumerate(image_paths):
            try:
                features = self.extract_features(img_path)
                features_list.append(features)
                valid_paths.append(img_path)
                
                if progress_callback:
                    progress = int((idx + 1) * 100 / total_images)
                    progress_callback(progress)
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
        
        return features_list, valid_paths

    def __del__(self):
        # Restore original forward method when object is destroyed
        if hasattr(self, 'original_forward'):
            self.model.forward = self.original_forward