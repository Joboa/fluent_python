from abc import ABC, abstractmethod

class TrainingStrategy(ABC):
    """Abstract training strategy"""
    
    @abstractmethod
    def train_step(self, model, batch, optimizer):
        pass
    
    @abstractmethod
    def val_step(self, model, batch):
        pass

class SupervisedTraining(TrainingStrategy):
    """Standard supervised training"""
    # TODO: Implement

class SelfSupervisedTraining(TrainingStrategy):
    """Self-supervised training for autoencoders"""
    # TODO: Implement

class AdversarialTraining(TrainingStrategy):
    """Adversarial training strategy"""
    # TODO: Implement

# Usage: trainer can work with any strategy