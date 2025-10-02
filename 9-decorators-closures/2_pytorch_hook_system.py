import torch
import torch.nn as nn

def activation_hook(name):
    """Create hook function to capture intermediate activations"""
    
    activations = {}
    
    def hook(module, input, output):
        # TODO: Store activation in closure variable
        # This demonstrates closures in practice
        pass
    
    return hook, activations

class ModelAnalyzer:
    """Analyze model internals using hooks"""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
    
    def register_hooks(self):
        """Register hooks on all conv layers"""
        # TODO: Use the activation_hook function above
        pass
    
    def get_activation_stats(self):
        """Get statistics about activations"""
        # TODO: Compute mean, std, sparsity for each layer
        pass

# Test with a simple CNN