"""
CHALLENGE: Implement a Dataset class that's more sophisticated than PyTorch's.
"""

class SmartDataset:
    """
    Enterprise-grade dataset with advanced features.
    
    Requirements:
    1. Support multiple indexing patterns
    2. Lazy loading with caching
    3. Automatic batching
    4. Memory mapping for large datasets
    5. Thread-safe access
    """
    
    def __init__(self, data_source, transform=None, cache_size=1000):
        """
        TODO: Design for datasets that don't fit in memory.
        Consider: How do you balance memory usage vs access speed?
        """
        pass
    
    def __len__(self) -> int:
        """Should work even for streaming datasets"""
        pass
    
    def __getitem__(self, key) -> Union[Any, 'Batch']:
        """
        Support multiple access patterns:
        - dataset[0] -> single item
        - dataset[0:10] -> batch of 10 items  
        - dataset[[1,3,5]] -> fancy indexing
        - dataset[mask] -> boolean indexing
        """
        pass
    
    def __iter__(self):
        """Memory-efficient iteration over large datasets"""
        pass
    
    def __contains__(self, item) -> bool:
        """Fast membership testing"""
        pass
    
    # Advanced features
    def batch(self, size: int) -> 'BatchedDataset':
        """Return a view that yields batches"""
        pass
    
    def shuffle(self, seed=None) -> 'SmartDataset':
        """Shuffled view without copying data"""
        pass
    
    def filter(self, predicate) -> 'SmartDataset':
        """Filtered view with lazy evaluation"""
        pass

# INTERVIEW QUESTION STYLE:
"""
Explain the trade-offs between:
1. __getitem__ returning individual items vs batches
2. Eager loading vs lazy loading 
3. Memory mapping vs RAM caching
4. Thread safety vs performance

How would your design change for:
- Streaming data (infinite datasets)
- Multi-modal data (images + text)  
- Distributed training scenarios
"""