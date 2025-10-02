class ExperimentConfig:
    """Dynamic configuration with validation"""
    
    def __init__(self, **kwargs):
        # TODO: Store config values
        pass
    
    def __setattr__(self, name, value):
        """Validate config values on assignment"""
        # TODO: Add validation logic
        # e.g., learning_rate must be > 0
        super().__setattr__(name, value)
    
    def __getattr__(self, name):
        """Provide defaults for missing config"""
        defaults = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"No config for '{name}'")
    
    def update(self, **kwargs):
        """Update multiple config values"""
        # TODO: Implement batch update
        pass

# Usage:
# config = ExperimentConfig()
# config.learning_rate = 0.01  # Validates
# print(config.batch_size)     # Uses default