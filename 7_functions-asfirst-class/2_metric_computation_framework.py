def metric_factory(metric_name):
    """Return appropriate metric function based on name"""
    
    def accuracy(y_true, y_pred):
        # TODO: Implement
        pass
    
    def precision(y_true, y_pred):
        # TODO: Implement
        pass
    
    def f1_score(y_true, y_pred):
        # TODO: Implement
        pass
    
    # TODO: Return the correct function based on metric_name
    # This demonstrates functions as first-class objects

# Advanced: Create a metric that combines multiple functions
def combined_metric(*metric_functions):
    """Combine multiple metrics into one function"""
    # TODO: Return function that applies all metrics
    pass