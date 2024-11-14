
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


import os
import json
import torch
import faiss
import numpy as np
from typing import List, Tuple, Optional, Callable
from datetime import datetime
import logging
from vit_image_retrieval.core.feature_extractor import ImageFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageRetrievalSystem:
    def __init__(self, 
                 feature_extractor: Optional[ImageFeatureExtractor] = None,
                 index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 use_gpu: bool = False):
        """
        Initialize the retrieval system.
        
        Args:
            feature_extractor: Optional pre-initialized feature extractor
            index_path: Path to existing FAISS index
            metadata_path: Path to existing metadata
            use_gpu: Whether to use GPU for FAISS operations
        """
        self.feature_extractor = feature_extractor or ImageFeatureExtractor()
        self.feature_dim = self.feature_extractor.feature_dim
        logger.info(f"Initializing retrieval system with dimension: {self.feature_dim}")
        
        # Initialize FAISS index with inner product similarity
        self.index = None  # We'll initialize it in create_new_index() or load_index()
        self.metadata = {}
        
        # Store GPU preference
        self.use_gpu = use_gpu
        
        # If paths provided, load existing index
        if index_path and metadata_path:
            self.load(index_path, metadata_path)
        else:
            # Create new empty index
            self.create_new_index()
            
    def create_new_index(self):
        """Create a new empty FAISS index."""
        self.index = faiss.IndexFlatIP(self.feature_dim)
        
        # Move to GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Reset metadata
        self.metadata = {}
        
        logger.info("Created new empty FAISS index")

    def index_images(self, 
                    image_dir: str, 
                    progress_callback: Optional[Callable] = None) -> None:
        """
        Index all images in the specified directory.
        
        Args:
            image_dir: Directory containing images to index
            progress_callback: Optional callback function to report progress
        """
        logger.info(f"Indexing images from {image_dir}")
        
        # Create new index for the new set of images
        self.create_new_index()
        
        # Extract features using the feature extractor
        features_list, valid_paths = self.feature_extractor.extract_batch_features(
            image_dir, 
            progress_callback
        )
        
        if not features_list:
            raise ValueError("No valid features extracted from images")
            
        # Combine all features
        all_features = np.stack(features_list)
        logger.info(f"Feature array shape: {all_features.shape}")
        logger.info(f"Feature stats - Min: {all_features.min():.4f}, Max: {all_features.max():.4f}")
        
        # Add to index
        self.index.add(all_features)
        logger.info(f"Total vectors in index: {self.index.ntotal}")
        
        # Update metadata with additional information
        for idx, path in enumerate(valid_paths):
            self.metadata[str(idx)] = {
                'path': path,
                'filename': os.path.basename(path),
                'indexed_at': datetime.now().isoformat(),
                'feature_stats': {
                    'min': float(np.min(features_list[idx])),
                    'max': float(np.max(features_list[idx])),
                    'mean': float(np.mean(features_list[idx])),
                    'norm': float(np.linalg.norm(features_list[idx]))
                }
            }
        
        logger.info(f"Successfully indexed {len(valid_paths)} images")

    def search(self, 
              query_image_path: str,
              k: int = 5,
              distance_threshold: float = float('inf')) -> List[Tuple[str, float]]:
        """
        Search for similar images.
        
        Args:
            query_image_path: Path to query image
            k: Number of results to return
            distance_threshold: Maximum allowed distance for matches
            
        Returns:
            List of tuples containing (image_path, similarity_score)
        """
        logger.info(f"Searching for similar images to {query_image_path}")
        logger.info(f"Total images in index: {self.index.ntotal}")
        
        # Extract features from query image
        query_features = self.feature_extractor.extract_features(query_image_path)
        logger.info(f"Query feature shape: {query_features.shape}")
        
        # Search index
        k = min(k, self.index.ntotal)  # Make sure k doesn't exceed number of indexed images
        distances, indices = self.index.search(
            query_features.reshape(1, -1),
            k
        )
        
        logger.info(f"Raw search results - distances: {distances[0]}")
        logger.info(f"Raw search results - indices: {indices[0]}")
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            str_idx = str(int(idx))
            if str_idx in self.metadata and dist <= distance_threshold:
                image_info = self.metadata[str_idx]
                similarity = self._compute_similarity_score(dist)
                results.append((
                    image_info['path'],
                    float(similarity),
                    {
                        'distance': float(dist),
                        'filename': image_info['filename'],
                        'indexed_at': image_info['indexed_at'],
                        'feature_stats': image_info['feature_stats']
                    }
                ))
                logger.info(f"Match found: {image_info['path']} with similarity {similarity:.3f}")
            else:
                logger.debug(f"Skipping index {idx} (distance: {dist:.3f})")
        
        # Sort results by similarity (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        
        if not results:
            logger.warning("No matches found!")
        else:
            logger.info(f"Found {len(results)} matches")
            
        return results

    def _compute_similarity_score(self, distance: float) -> float:
        """
        Convert distance to a similarity score in [0, 1] range.
        Higher score means more similar.
        """
        # Since we're using cosine similarity (inner product with normalized vectors),
        # the distance is already in [-1, 1] range
        # Convert to [0, 1] range where 1 is most similar
        return (distance + 1) / 2

    def save(self, index_path: str, metadata_path: str) -> None:
        """Save the index and metadata to disk."""
        # If index is on GPU, move it back to CPU first
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        logger.info(f"Saved index with {self.index.ntotal} vectors")
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load the index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Move to GPU if needed
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            # Verify dimension matches
            if self.index.d != self.feature_dim:
                raise ValueError(
                    f"Index dimension ({self.index.d}) does not match "
                    f"feature extractor dimension ({self.feature_dim})"
                )
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            logger.info(f"Metadata contains {len(self.metadata)} entries")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            'num_images': self.index.ntotal,
            'feature_dimension': self.feature_dim,
            'gpu_enabled': self.use_gpu,
            'metadata_entries': len(self.metadata)
        }

    def __del__(self):
        """Cleanup resources."""
        # If using GPU, explicitly delete the index to free GPU memory
        if hasattr(self, 'index') and self.use_gpu:
            del self.index