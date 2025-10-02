import torch.nn as nn

# BAD: Deep inheritance hierarchy
class Model(nn.Module): pass
class ConvModel(Model): pass
class ResNetLike(ConvModel): pass
class MySpecificModel(ResNetLike): pass  # Hard to maintain!

# GOOD: Composition-based design
class ConvBlock:
    """Reusable convolutional block"""
    # TODO: Implement as composable component

class AttentionBlock:
    """Reusable attention mechanism"""
    # TODO: Implement

class FlexibleModel(nn.Module):
    """Model built from composable blocks"""
    
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        # TODO: Process through all blocks
        pass

# Usage:
# model = FlexibleModel([ConvBlock(...), AttentionBlock(...), ConvBlock(...)])