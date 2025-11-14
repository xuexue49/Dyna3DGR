"""
K-Nearest Neighbors (KNN) search utilities.

Provides efficient KNN search for finding nearest control nodes.
"""

import torch
from typing import Tuple, Optional


def knn_search(
    query_points: torch.Tensor,  # [M, 3]
    reference_points: torch.Tensor,  # [N, 3]
    k: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest neighbors for each query point.
    
    Uses PyTorch's built-in operations for efficiency.
    
    Args:
        query_points: Query point positions [M, 3]
        reference_points: Reference point positions [N, 3]
        k: Number of nearest neighbors
    
    Returns:
        indices: [M, k] indices of nearest neighbors
        distances: [M, k] distances to nearest neighbors
    """
    M = query_points.shape[0]
    N = reference_points.shape[0]
    
    # Ensure k is valid
    k = min(k, N)
    
    # Compute pairwise squared distances [M, N]
    # ||q - r||² = ||q||² + ||r||² - 2<q, r>
    query_sq = torch.sum(query_points ** 2, dim=-1, keepdim=True)  # [M, 1]
    ref_sq = torch.sum(reference_points ** 2, dim=-1, keepdim=True)  # [N, 1]
    dot_product = torch.mm(query_points, reference_points.t())  # [M, N]
    
    distances_sq = query_sq + ref_sq.t() - 2 * dot_product  # [M, N]
    
    # Clamp to avoid numerical issues
    distances_sq = torch.clamp(distances_sq, min=0.0)
    
    # Find k smallest distances
    distances_sq_topk, indices = torch.topk(
        distances_sq,
        k=k,
        dim=1,
        largest=False,
        sorted=True,
    )
    
    distances = torch.sqrt(distances_sq_topk)
    
    return indices, distances


def knn_search_batch(
    query_points: torch.Tensor,  # [B, M, 3]
    reference_points: torch.Tensor,  # [B, N, 3]
    k: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched KNN search.
    
    Args:
        query_points: Query point positions [B, M, 3]
        reference_points: Reference point positions [B, N, 3]
        k: Number of nearest neighbors
    
    Returns:
        indices: [B, M, k] indices of nearest neighbors
        distances: [B, M, k] distances to nearest neighbors
    """
    B, M, _ = query_points.shape
    N = reference_points.shape[1]
    
    # Ensure k is valid
    k = min(k, N)
    
    # Compute pairwise squared distances [B, M, N]
    query_sq = torch.sum(query_points ** 2, dim=-1, keepdim=True)  # [B, M, 1]
    ref_sq = torch.sum(reference_points ** 2, dim=-1, keepdim=True)  # [B, N, 1]
    dot_product = torch.bmm(query_points, reference_points.transpose(1, 2))  # [B, M, N]
    
    distances_sq = query_sq + ref_sq.transpose(1, 2) - 2 * dot_product  # [B, M, N]
    
    # Clamp to avoid numerical issues
    distances_sq = torch.clamp(distances_sq, min=0.0)
    
    # Find k smallest distances
    distances_sq_topk, indices = torch.topk(
        distances_sq,
        k=k,
        dim=2,
        largest=False,
        sorted=True,
    )
    
    distances = torch.sqrt(distances_sq_topk)
    
    return indices, distances


# Try to import PyTorch3D for more efficient KNN
try:
    from pytorch3d.ops import knn_points
    
    def knn_search_pytorch3d(
        query_points: torch.Tensor,  # [M, 3]
        reference_points: torch.Tensor,  # [N, 3]
        k: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        KNN search using PyTorch3D's optimized implementation.
        
        Args:
            query_points: Query point positions [M, 3]
            reference_points: Reference point positions [N, 3]
            k: Number of nearest neighbors
        
        Returns:
            indices: [M, k] indices of nearest neighbors
            distances: [M, k] distances to nearest neighbors
        """
        # Add batch dimension
        query = query_points.unsqueeze(0)  # [1, M, 3]
        ref = reference_points.unsqueeze(0)  # [1, N, 3]
        
        # KNN search
        knn_result = knn_points(query, ref, K=k, return_nn=False)
        
        # Remove batch dimension
        indices = knn_result.idx.squeeze(0)  # [M, k]
        distances = torch.sqrt(knn_result.dists.squeeze(0))  # [M, k]
        
        return indices, distances
    
    # Use PyTorch3D implementation if available
    _knn_search_impl = knn_search_pytorch3d
    _has_pytorch3d = True

except ImportError:
    _knn_search_impl = knn_search
    _has_pytorch3d = False


def get_knn_implementation() -> str:
    """
    Get the current KNN implementation being used.
    
    Returns:
        implementation: 'pytorch3d' or 'pytorch'
    """
    return 'pytorch3d' if _has_pytorch3d else 'pytorch'


def knn_search_auto(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Automatically select the best available KNN implementation.
    
    Uses PyTorch3D if available, otherwise falls back to PyTorch implementation.
    
    Args:
        query_points: Query point positions [M, 3]
        reference_points: Reference point positions [N, 3]
        k: Number of nearest neighbors
    
    Returns:
        indices: [M, k] indices of nearest neighbors
        distances: [M, k] distances to nearest neighbors
    """
    return _knn_search_impl(query_points, reference_points, k)


class KNNCache:
    """
    Cache for KNN search results to avoid redundant computation.
    
    Useful when control node positions don't change frequently.
    """
    
    def __init__(self, max_cache_size: int = 100):
        """
        Initialize KNN cache.
        
        Args:
            max_cache_size: Maximum number of cached results
        """
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_order = []
    
    def get(
        self,
        query_points: torch.Tensor,
        reference_points: torch.Tensor,
        k: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached KNN result if available.
        
        Args:
            query_points: Query point positions
            reference_points: Reference point positions
            k: Number of nearest neighbors
        
        Returns:
            result: Cached (indices, distances) or None
        """
        # Create cache key
        key = (
            id(query_points),
            id(reference_points),
            k,
        )
        
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        return None
    
    def put(
        self,
        query_points: torch.Tensor,
        reference_points: torch.Tensor,
        k: int,
        result: Tuple[torch.Tensor, torch.Tensor],
    ):
        """
        Cache KNN result.
        
        Args:
            query_points: Query point positions
            reference_points: Reference point positions
            k: Number of nearest neighbors
            result: (indices, distances) to cache
        """
        # Create cache key
        key = (
            id(query_points),
            id(reference_points),
            k,
        )
        
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Add to cache
        self.cache[key] = result
        self.access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


def knn_search_with_cache(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    k: int = 4,
    cache: Optional[KNNCache] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KNN search with optional caching.
    
    Args:
        query_points: Query point positions [M, 3]
        reference_points: Reference point positions [N, 3]
        k: Number of nearest neighbors
        cache: Optional KNN cache
    
    Returns:
        indices: [M, k] indices of nearest neighbors
        distances: [M, k] distances to nearest neighbors
    """
    if cache is not None:
        # Try to get from cache
        result = cache.get(query_points, reference_points, k)
        if result is not None:
            return result
    
    # Compute KNN
    indices, distances = knn_search_auto(query_points, reference_points, k)
    
    if cache is not None:
        # Store in cache
        cache.put(query_points, reference_points, k, (indices, distances))
    
    return indices, distances
