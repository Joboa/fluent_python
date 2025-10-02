import numpy as np

# Create sample data representing feature maps
feature_maps = np.random.randn(32, 64, 28, 28)  # batch, channels, height, width

# TODO: Using array slicing and comprehensions:
# 1. Extract every other channel
# 2. Downsample by factor of 2 (every other pixel)
# 3. Calculate mean activation per channel
# 4. Find channels with activation > threshold

# Compare performance: list comprehension vs generator vs NumPy vectorization