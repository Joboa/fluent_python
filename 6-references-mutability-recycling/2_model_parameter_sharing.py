import torch.nn as nn

class SharedWeightModel(nn.Module):
    """Model that shares weights between layers"""
    
    def __init__(self):
        super().__init__()
        # TODO: Create a conv layer and reuse it multiple times
        # Be careful about parameter sharing vs copying!
        pass

# Test: Do gradients accumulate properly with shared weights?