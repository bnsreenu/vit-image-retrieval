
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
                 use_gpu: bool = False,
                 n_regions: int = None,  # New parameter
                 nprobe: int = None):    # New parameter
        """
        Initialize the retrieval system.
        
        Args:
            feature_extractor: Optional pre-initialized feature extractor
            index_path: Path to existing FAISS index
            metadata_path: Path to existing metadata
            use_gpu: Whether to use GPU for FAISS operations
            n_regions: Number of IVF clusters (calculated automatically if None)
            nprobe: Number of regions to search (calculated automatically if None)
        """
        self.feature_extractor = feature_extractor or ImageFeatureExtractor()
        self.feature_dim = self.feature_extractor.feature_dim
        self.use_gpu = use_gpu
        self.n_regions = n_regions
        self.nprobe = nprobe
        self.is_trained = False
        
        logger.info(f"Initializing retrieval system with dimension: {self.feature_dim}")
        
        self.index = None  # We'll initialize it in create_new_index() or load_index()
        self.metadata = {}
        
        # If paths provided, load existing index
        if index_path and metadata_path:
            self.load(index_path, metadata_path)
        else:
            # Create new empty index
            self.create_new_index()
            
    def calculate_optimal_regions(self, num_vectors: int) -> Tuple[int, int]:
        """
        Calculate optimal number of regions and probes based on dataset size.
        
        Rules of thumb:
        - For very small datasets (<100): use 10% of dataset size
        - For small datasets (<1000): use sqrt(N)
        - For larger datasets: use 4*sqrt(N), but cap at N/10
        """
        if num_vectors < 100:
            n_regions = max(1, num_vectors // 10)  # 10% of dataset size
        elif num_vectors < 1000:
            n_regions = int(np.sqrt(num_vectors))
        else:
            n_regions = min(int(4 * np.sqrt(num_vectors)), num_vectors // 10)
            
        # Make sure n_regions is smaller than dataset size
        n_regions = min(n_regions, num_vectors - 1)
        
        # Calculate number of probes (regions to search)
        nprobe = max(1, min(n_regions // 4, 20))  # Cap at 20 probes
        
        logger.info(f"Dataset size: {num_vectors}")
        logger.info(f"Using {n_regions} regions and {nprobe} probes")
        
        return n_regions, nprobe
    
    def create_new_index(self, num_vectors: int = None):
        """Create a new empty FAISS IVF index."""
        # Calculate n_regions if not specified
        if num_vectors is not None:
            self.n_regions, self.nprobe = self.calculate_optimal_regions(num_vectors)
        elif self.n_regions is None:
            # Default conservative values for unknown size
            self.n_regions = 10  # Very conservative default
            self.nprobe = 4     # Search 40% of regions
                
        logger.info(f"Creating IVF index with {self.n_regions} regions")
        
        # Create quantizer and IVF index
        self.quantizer = faiss.IndexFlatIP(self.feature_dim)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, 
            self.feature_dim, 
            self.n_regions,
            faiss.METRIC_INNER_PRODUCT
        )
        
        # Set number of regions to search
        self.index.nprobe = self.nprobe
        logger.info(f"Will search {self.nprobe} regions during queries")
        
        # Move to GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Reset metadata
        self.metadata = {}
        self.is_trained = False
        
        logger.info("Created new empty FAISS IVF index")
    
    def index_images(self, 
                    image_dir: str, 
                    progress_callback: Optional[Callable] = None) -> None:
        """Index all images in the specified directory."""
        logger.info(f"Indexing images from {image_dir}")
        
        # Extract features using the feature extractor
        features_list, valid_paths = self.feature_extractor.extract_batch_features(
            image_dir, 
            progress_callback
        )
        
        if not features_list:
            raise ValueError("No valid features extracted from images")
            
        # Combine all features
        all_features = np.stack(features_list)
        num_vectors = len(features_list)
        
        logger.info(f"Extracted features from {num_vectors} images")
        
        # Create new index with appropriate size
        self.create_new_index(num_vectors=num_vectors)
        
        # Train the index first
        logger.info("Training IVF index...")
        logger.info(f"Training with {num_vectors} vectors and {self.n_regions} regions")
        self.index.train(all_features)
        self.is_trained = True
        
        # Add vectors to index
        self.index.add(all_features)
        logger.info(f"Added {num_vectors} vectors to index")
    
        
        # Update metadata
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

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load the index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Set important IVF attributes
            if isinstance(self.index, faiss.IndexIVFFlat):
                self.n_regions = self.index.nlist  # Get number of regions from loaded index
                self.nprobe = self.index.nprobe    # Get number of probes
                self.is_trained = True             # Loaded IVF index is already trained
            
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
            
            logger.info(f"Loaded IVF index with {self.index.ntotal} vectors")
            logger.info(f"Number of regions (nlist): {self.n_regions}")
            logger.info(f"Number of probes (nprobe): {self.nprobe}")
            logger.info(f"Metadata contains {len(self.metadata)} entries")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def _compute_similarity_score(self, distance: float) -> float:
        """
        Convert inner product distance to a similarity score in [0, 1] range.
        For IVF index using inner product, higher is better.
        """
        # Since we're using inner product with normalized vectors,
        # the similarity is already in [-1, 1] range
        # Convert to [0, 1] range where 1 is most similar
        return (distance + 1) / 2            
            

    def search(self, 
              query_image_path: str,
              k: int = 5,
              distance_threshold: float = float('inf')) -> List[Tuple[str, float]]:
        """Search for similar images."""
        logger.info(f"Searching for similar images to {query_image_path}")
        logger.info(f"Total images in index: {self.index.ntotal}")
        
        try:
            # Extract features from query image
            query_features = self.feature_extractor.extract_features(query_image_path)
            
            # Ensure features are in the correct format
            query_features = query_features.astype(np.float32)  # FAISS requires float32
            query_features = query_features.reshape(1, -1)  # Ensure 2D array shape
            
            logger.info(f"Query feature shape: {query_features.shape}")
            logger.info(f"Query feature type: {query_features.dtype}")
            
            # Verify features are not None and have correct shape
            if query_features is None or query_features.size == 0:
                raise ValueError("Failed to extract features from query image")
                
            # Search index
            k = min(k, self.index.ntotal)  # Make sure k doesn't exceed number of indexed images
            distances, indices = self.index.search(query_features, k)
            
            logger.info(f"Raw search results - distances: {distances[0]}")
            logger.info(f"Raw search results - indices: {indices[0]}")
            
            # Prepare results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                str_idx = str(int(idx))
                if str_idx in self.metadata and dist <= distance_threshold:
                    image_info = self.metadata[str_idx]
                    similarity = self._compute_similarity_score(float(dist))
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
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            logger.error(f"Index type: {type(self.index)}")
            logger.error(f"Index properties: ntotal={self.index.ntotal}, d={self.index.d}")
            raise RuntimeError(f"Search failed: {str(e)}")
        

    def save(self, index_path: str, metadata_path: str) -> None:
        """Save the index and metadata to disk."""
        # If index is on GPU, convert back to CPU for saving
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        logger.info(f"Saved IVF index with {self.index.ntotal} vectors")
        logger.info(f"Using {self.n_regions} regions, probing {self.nprobe}")
        logger.info(f"Saved to {index_path} and {metadata_path}")

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            'num_images': self.index.ntotal,
            'feature_dimension': self.feature_dim,
            'gpu_enabled': self.use_gpu,
            'metadata_entries': len(self.metadata),
            'n_regions': self.n_regions,
            'nprobe': self.nprobe if hasattr(self, 'index') else None,
            'is_trained': self.is_trained
        }

    def __del__(self):
        """Cleanup resources."""
        # If using GPU, explicitly delete the index to free GPU memory
        if hasattr(self, 'index') and self.use_gpu:
            del self.index