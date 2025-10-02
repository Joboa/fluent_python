# Given: list of image paths and corresponding labels
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
labels = [0, 1, 0, 1]

# TODO: Using list comprehensions, create:
# 1. List of (path, label) tuples for training data
# 2. Separate lists for each class
# 3. Batch the data into groups of 2
# 4. Create a validation split (20% of data)

# Advanced: Use generator expressions for memory efficiency
def data_generator(paths, labels, batch_size):
    """Generate batches without loading all data into memory"""
    # TODO: Implement using yield
    pass