class SimpleTensor:
    """Build a basic tensor class using Python's data model"""
    
    def __init__(self, data):
        # TODO: Store data and shape
        pass
    
    def __len__(self):
        # TODO: Return number of elements
        pass
    
    def __getitem__(self, key):
        # TODO: Support indexing and slicing
        pass
    
    def __repr__(self):
        # TODO: Pretty printing
        pass
    
    def __add__(self, other):
        # TODO: Element-wise addition
        pass

# Test your implementation
tensor = SimpleTensor([[1, 2], [3, 4]])
print(len(tensor))  # Should work
print(tensor[0])    # Should work
print(tensor + tensor)  # Should work