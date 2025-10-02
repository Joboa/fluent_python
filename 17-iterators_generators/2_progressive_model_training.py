def progressive_training(model, data_generator, validation_data):
    """Training with progressive validation"""
    
    epoch = 0
    best_loss = float('inf')
    
    # TODO: Use generators to:
    # 1. Stream training data
    # 2. Periodically yield validation metrics
    # 3. Implement early stopping
    # 4. Handle infinite data streams
    
    for batch in data_generator:
        # Training step...
        
        if should_validate():  # Every N steps
            val_loss = yield validate(model, validation_data)  # Generator coroutine!
            if val_loss < best_loss:
                best_loss = val_loss
                # Save model...

# This is a coroutine-based training loop!