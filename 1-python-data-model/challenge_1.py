"""
CHALLENGE: Build a tensor library that could be the foundation of PyTorch.
Your implementation will be evaluated on:
- Memory efficiency
- API design
- Performance characteristics  
- Code readability
"""

import numpy as np
from typing import Tuple, Union, Optional, Any
import weakref

class Tensor:
    """
    Production-grade tensor implementation.
    
    Requirements:
    1. Support n-dimensional arrays with efficient memory layout
    2. Implement automatic broadcasting like NumPy
    3. Support method chaining (fluent interface)
    4. Handle memory sharing correctly
    5. Integrate with Python's data model seamlessly
    """
    
    def __init__(self, data, dtype=None, requires_grad=False):
        """
        TODO: Initialize tensor with:
        - Efficient memory layout (C-contiguous vs Fortran)
        - Type system that prevents common bugs
        - Gradient tracking infrastructure
        
        Consider: How does PyTorch handle these decisions?
        """
        pass
    
    def __len__(self) -> int:
        """Return size of first dimension (like PyTorch)"""
        pass
    
    def __getitem__(self, key) -> 'Tensor':
        """
        Support advanced indexing:
        - Single indices: tensor[0]
        - Slices: tensor[1:3]  
        - Multi-dimensional: tensor[1:3, :, 2]
        - Boolean masking: tensor[tensor > 0]
        - Fancy indexing: tensor[[1, 3, 5]]
        """
        pass
    
    def __setitem__(self, key, value):
        """In-place modification with gradient tracking"""
        pass
    
    def __repr__(self) -> str:
        """PyTorch-style pretty printing with shape info"""
        pass
    
    def __add__(self, other) -> 'Tensor':
        """Broadcasting addition with gradient support"""
        pass
    
    def __iadd__(self, other) -> 'Tensor':
        """In-place addition - critical for memory optimization"""
        pass
    
    def __matmul__(self, other) -> 'Tensor':
        """Matrix multiplication with optimal algorithm selection"""
        pass
    
    # Properties that PyTorch tensors have
    @property
    def shape(self) -> Tuple[int, ...]:
        pass
    
    @property 
    def dtype(self):
        pass
    
    @property
    def device(self):
        pass
    
    def view(self, *shape) -> 'Tensor':
        """Memory view without copying data"""
        pass
    
    def clone(self) -> 'Tensor':
        """Deep copy with gradient graph preservation"""
        pass

# EVALUATION CRITERIA:
# 1. Does your tensor integrate seamlessly with Python (len, iter, indexing)?
# 2. Can you explain the memory layout decisions?
# 3. How does broadcasting work in your implementation?
# 4. What are the performance implications of each magic method?

# TEST YOUR UNDERSTANDING:
def test_tensor_api():
    """Write tests that would break a naive implementation"""
    t1 = Tensor([[1, 2], [3, 4]])
    t2 = Tensor([10, 20])
    
    # These should all work seamlessly:
    assert len(t1) == 2
    assert t1[0].shape == (2,)
    result = t1 + t2  # Broadcasting
    assert result.shape == (2, 2)
    
    # Memory sharing test:
    view = t1.view(-1)  # Reshape
    original_ptr = id(t1._data)  # Internal data pointer
    view_ptr = id(view._data)
    assert original_ptr == view_ptr, "Views should share memory"
    
    # In-place operations:
    original_id = id(t1)
    t1 += 1
    assert id(t1) == original_id, "In-place ops shouldn't create new objects"