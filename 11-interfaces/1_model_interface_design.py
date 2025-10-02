from abc import ABC, abstractmethod
from typing import Protocol
import torch

class ModelProtocol(Protocol):
    """Protocol for all ML models"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        ...
    
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        ...

class BaseAutoencoder(ABC):
    """Abstract base class for autoencoders"""
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward implementation"""
        z = self.encode(x)
        return self.decode(z)

# TODO: Implement ConvolutionalAutoencoder(BaseAutoencoder)
# TODO: Implement VariationalAutoencoder(BaseAutoencoder)