import torch

def memory_quiz():
    """Understanding tensor memory sharing"""
    
    # TODO: Predict what happens to memory usage:
    a = torch.randn(1000, 1000)
    b = a.view(-1)  # Reshape
    c = a.clone()   # Copy
    d = a[::2]      # Slice
    
    # Which operations share memory? Which create copies?
    # Use a.data_ptr() to check memory addresses
    
    # In-place operations
    original_id = id(a)
    a += 1  # vs a = a + 1
    # What happened to the id?

def autograd_references():
    """Understanding how PyTorch tracks gradients"""
    
    # TODO: Explore gradient sharing
    x = torch.randn(5, 5, requires_grad=True)
    y = x.view(25)  # Share storage
    z = x.clone()   # Separate storage
    
    # How do gradients flow back?