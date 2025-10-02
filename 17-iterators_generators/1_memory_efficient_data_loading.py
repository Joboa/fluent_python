def infinite_data_generator(dataset_path, batch_size):
    """Generate infinite batches without memory buildup"""
    # TODO: Implement using yield
    # Should cycle through dataset forever
    pass

def streaming_predictions(model, data_stream):
    """Process streaming data with generators"""
    # TODO: Yield predictions one at a time
    # Handle backpressure gracefully
    pass

class LazyDataset:
    """Dataset that loads data on-demand"""
    
    def __init__(self, file_list):
        self.files = file_list
    
    def __iter__(self):
        """Make dataset iterable"""
        # TODO: Yield data samples one at a time
        pass
    
    def batch_iter(self, batch_size):
        """Generate batches lazily"""
        # TODO: Implement batching with generators
        pass

# Memory usage should stay constant regardless of dataset size!