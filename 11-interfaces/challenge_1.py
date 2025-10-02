# """
# CHALLENGE: Design a model framework that supports:
# 1. Research experimentation (easy to extend)
# 2. Production deployment (optimized inference)  
# 3. Multi-modal models (vision + NLP)
# 4. Distributed training (model parallelism)
# 5. Hardware optimization (mobile, TPU, GPU)
# """

# from abc import ABC, abstractmethod
# from typing import Protocol, Dict, Any, Optional, Union, Tuple
# import torch
# import torch.nn as nn
# from dataclasses import dataclass
# from enum import Enum

# class ModelProtocol(Protocol):
#     """Protocol defining what all models must support"""
    
#     def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Forward pass with flexible input/output"""
#         ...
    
#     def get_loss(self, 
#                 predictions: Dict[str, torch.Tensor], 
#                 targets: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """Compute loss for training"""
#         ...
    
#     def get_metrics(self, 
#                    predictions: Dict[str, torch.Tensor], 
#                    targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
#         """Compute evaluation metrics"""
#         ...
    
#     @property
#     def num_parameters(self) -> int:
#         """Total parameter count"""
#         ...

# class BaseModel(ABC, nn.Module):
#     """
#     TODO: Abstract base class for all models in the platform.
    
#     Provides common functionality while enforcing interface.
#     """
    
#     def __init__(self, config: 'ModelConfig'):
#         super().__init__()
#         self.config = config
    
#     @abstractmethod
#     def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """Encode inputs to latent representation"""
#         pass
    
#     @abstractmethod
#     def decode(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """Decode latents to outputs"""
#         pass
    
#     def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Default implementation using encode/decode"""
#         latents = self.encode(inputs)
#         return self.decode(latents)
    
#     # Common functionality for all models
#     def save_checkpoint(self, path: str):
#         """TODO: Implement standard checkpointing"""
#         pass
    
#     def load_checkpoint(self, path: str):
#         """TODO: Implement checkpoint loading with validation"""
#         pass
    
#     def optimize_for_inference(self) -> 'InferenceModel':
#         """TODO: Convert to optimized inference version"""
#         pass
    
#     def get_computational_graph_info(self) -> Dict[str, Any]:
#         """TODO: Return FLOPs, memory usage, etc."""
#         pass

# @dataclass
# class ModelConfig:
#     """Configuration for model instantiation"""
#     model_type: str
#     hidden_size: int
#     num_layers: int
#     dropout: float = 0.1
#     activation: str = 'relu'
#     # TODO: Add validation and type checking
    
#     def validate(self):
#         """TODO: Validate configuration parameters"""
#         pass

# class MultiModalModel(BaseModel):
#     """
#     TODO: Base class for models handling multiple input modalities.
    
#     Examples: CLIP (vision + text), LayoutLM (text + layout), etc.
#     """
    
#     def __init__(self, config: ModelConfig):
#         super().__init__(config)
#         self.modality_encoders = nn.ModuleDict()
#         self.fusion_layer = None  # TODO: Implement fusion strategies
    
#     def add_modality_encoder(self, modality: str, encoder: nn.Module):
#         """TODO: Register encoder for specific modality"""
#         pass
    
#     def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """TODO: Encode multiple modalities and fuse"""
#         pass

# class DistributedModel(BaseModel):
#     """
#     TODO: Model that can be split across multiple devices/nodes.
    
#     Requirements:
#     1. Automatic partitioning based on memory constraints
#     2. Pipeline parallelism support
#     3. Gradient synchronization
#     4. Fault tolerance for node failures
#     """
    
#     def __init__(self, config: ModelConfig, device_map: Dict[str, str]):
#         super().__init__(config)
#         self.device_map = device_map
#         self.partition_points = []  # Where to split the model
    
#     def partition_model(self, memory_budget: Dict[str, int]):
#         """TODO: Automatically partition model across devices"""
#         pass
    
#     def forward_with_pipeline(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """TODO: Pipeline parallel forward pass"""
#         pass

# # Factory pattern for model creation
# class ModelFactory:
#     """
#     TODO: Factory for creating models from configuration.
    
#     Supports:
#     1. Registration of custom model types
#     2. Automatic optimization based on target hardware
#     3. Configuration validation
#     4. Version compatibility checking
#     """
    
#     _registry: Dict[str, type] = {}
    
#     @classmethod
#     def register(cls, name: str, model_class: type):
#         """Register new model type"""
#         # TODO: Validate model_class implements required protocols
#         cls._registry[name] = model_class
    
#     @classmethod
#     def create(cls, config: ModelConfig) -> BaseModel:
#         """Create model instance from config"""
#         # TODO: Implement with validation and optimization
#         pass
    
#     @classmethod
#     def create_for_inference(cls, config: ModelConfig, 
#                            target_device: str = 'cpu') -> 'InferenceModel':
#         """Create optimized inference model"""
#         # TODO: Apply device-specific optimizations
#         pass

# # Plugin system for extensibility
# class ModelPlugin(ABC):
#     """Plugin interface for extending model capabilities"""
    
#     @abstractmethod
#     def apply(self, model: BaseModel) -> BaseModel:
#         """Apply plugin modifications to model"""
#         pass
    
#     @abstractmethod
#     def is_compatible(self, model_type: str) -> bool:
#         """Check if plugin is compatible with model type"""
#         pass

# class QuantizationPlugin(ModelPlugin):
#     """TODO: Plugin for model quantization"""
    
#     def apply(self, model: BaseModel) -> BaseModel:
#         # TODO: Apply INT8/FP16 quantization
#         pass

# class PruningPlugin(ModelPlugin):
#     """TODO: Plugin for structured/unstructured pruning"""
    
#     def apply(self, model: BaseModel) -> BaseModel:
#         # TODO: Apply pruning strategies
#         pass

# # ADVANCED CHALLENGE:
# class AdaptiveModel(BaseModel):
#     """
#     TODO: Model that adapts its architecture during training.
    
#     Examples:
#     - Progressive growing (start small, add layers)
#     - Neural architecture search integration
#     - Dynamic depth/width based on input complexity
#     """
    
#     def __init__(self, config: ModelConfig):
#         super().__init__(config)
#         self.growth_schedule = []  # When to add new components
#         self.complexity_predictor = None  # Predict input complexity
    
#     def grow_architecture(self, step: int):
#         """TODO: Add new layers/components based on schedule"""
#         pass
    
#     def adapt_to_input(self, inputs: Dict[str, torch.Tensor]) -> str:
#         """TODO: Choose architecture variant based on input"""
#         pass

# # INTERVIEW QUESTIONS:
# """
# 1. How would you extend this framework to support new model types?
# 2. What are the trade-offs between protocols vs abstract base classes?
# 3. How do you handle backward compatibility when model interfaces change?
# 4. Design a system for A/B testing different model architectures.
# 5. How would you implement model versioning and migration?
# """

# # SYSTEM DESIGN CHALLENGE:
# """
# You need to build a model serving system that:
# 1. Supports 100+ different model types
# 2. Can dynamically load models based on request routing
# 3. Provides automatic performance optimization
# 4. Handles model updates without downtime
# 5. Supports multi-tenant isolation

# How would you use protocols and ABCs to design this?
# """

# ---

# ### Chapter 12 - Inheritance vs Composition
# **🎯 Big Tech Question**: "Refactor this deep inheritance hierarchy. Explain your design choices."

# **Exercise 12.1: Architecture Refactoring Challenge**
# ```python
# """
# CHALLENGE: This is a real inheritance hierarchy that became unmaintainable.
# Refactor using composition patterns.
# """

# # BAD: Deep inheritance (typical in legacy ML codebases)
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dropout = 0.1
    
#     def forward(self, x):
#         raise NotImplementedError

# class ImageModel(Model):
#     def __init__(self):
#         super().__init__()
#         self.conv_base = nn.Sequential(...)
    
#     def extract_features(self, x):
#         return self.conv_base(x)

# class ClassificationModel(ImageModel):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.classifier = nn.Linear(512, num_classes)
    
#     def forward(self, x):
#         features = self.extract_features(x)
#         return self.classifier(features.mean(dim=(2,3)))

# class ResNetClassificationModel(ClassificationModel):
#     def __init__(self, num_classes, depth=50):
#         super().__init__(num_classes)
#         # Override parent's conv_base
#         self.conv_base = self._make_resnet(depth)
    
#     def _make_resnet(self, depth):
#         # ResNet implementation
#         pass

# class PretrainedResNetModel(ResNetClassificationModel):
#     def __init__(self, num_classes, pretrained_path):
#         super().__init__(num_classes)
#         self.load_pretrained(pretrained_path)
    
#     def load_pretrained(self, path):
#         # Loading logic
#         pass

# class FinetuningResNetModel(PretrainedResNetModel):
#     def __init__(self, num_classes, pretrained_path, freeze_backbone=True):
#         super().__init__(num_classes, pretrained_path)
#         if freeze_backbone:
#             self.freeze_backbone()
    
#     def freeze_backbone(self):
#         for param in self.conv_base.parameters():
#             param.requires_grad = False

# # PROBLEMS WITH ABOVE:
# # 1. Deep inheritance makes changes risky
# # 2. Tight coupling between components  
# # 3. Hard to mix and match features
# # 4. Difficult to test individual components
# # 5. New requirements require changing multiple classes

# # YOUR TASK: Refactor using composition
# class Block(ABC):
#     """Composable model block"""
    
#     @abstractmethod
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         pass
    
#     @abstractmethod
#     def get_output_size(self, input_size: Tuple[int, ...]) -> Tuple[int, ...]:
#         """Calculate output dimensions given input"""
#         pass

# class FeatureExtractor(Block):
#     """TODO: Composable feature extraction component"""
    
#     def __init__(self, architecture: str, pretrained: bool = False):
#         # TODO: Support different architectures (ResNet, EfficientNet, etc.)
#         pass

# class Classifier(Block):
#     """TODO: Composable classification head"""
    
#     def __init__(self, input_size: int, num_classes: int, dropout: float = 0.1):
#         pass

# class ModelComposer:
#     """
#     TODO: Compose models from reusable blocks.
    
#     Example usage:
#     model = ModelComposer()
#         .add_block(FeatureExtractor('resnet50', pretrained=True))
#         .add_block(GlobalAveragePooling())  
#         .add_block(Classifier(2048, 10))
#         .build()
#     """
    
#     def __init__(self):
#         self.blocks = []
    
#     def add_block(self, block: Block) -> 'ModelComposer':
#         """Fluent interface for adding blocks"""
#         # TODO: Implement with shape validation
#         pass
    
#     def build(self) -> nn.Module:
#         """Build final composed model"""
#         # TODO: Create sequential model with proper connections
#         pass

# class TrainingStrategy:
#     """TODO: Composable training strategies"""
    
#     def __init__(self, 
#                  freeze_policy: Optional['FreezePolicy'] = None,
#                  augmentation: Optional['AugmentationPolicy'] = None,
#                  regularization: Optional['RegularizationPolicy'] = None):
#         pass
    
#     def apply_to_model(self, model: nn.Module):
#         """Apply training strategy to model"""
#         pass

# class FreezePolicy(ABC):
#     """Strategy for freezing model parameters"""
    
#     @abstractmethod
#     def apply(self, model: nn.Module):
#         pass

# class BackboneFreezePolicy(FreezePolicy):
#     """TODO: Freeze backbone, train classifier only"""
#     pass

# class GradualUnfreezePolicy(FreezePolicy):
#     """TODO: Gradually unfreeze layers during training"""
#     pass

# # ADVANCED: Plugin Architecture
# class ModelPlugin(ABC):
#     """Plugin interface for extending model functionality"""
    
#     @abstractmethod
#     def modify_model(self, model: nn.Module) -> nn.Module:
#         pass

# class AttentionPlugin(ModelPlugin):
#     """TODO: Add attention mechanisms to existing models"""
#     pass

# class BatchNormPlugin(ModelPlugin):
#     """TODO: Add batch normalization where beneficial"""
#     pass

# # USAGE EXAMPLE:
# def create_flexible_model():
#     """
#     TODO: Demonstrate how composition makes experimentation easier.
    
#     Show how the same components can be recombined for:
#     1. Different architectures (ResNet vs EfficientNet backbone)
#     2. Different tasks (classification vs detection)
#     3. Different training strategies (from-scratch vs fine-tuning)
#     """
    
#     # Base model
#     base_model = ModelComposer()
#         .add_block(FeatureExtractor('efficientnet-b0'))
#         .add_block(GlobalAveragePooling())
#         .add_block(Classifier(1280, 10))
#         .build()
    
#     # Add plugins
#     attention_model = AttentionPlugin().modify_model(base_model)
    
#     # Apply training strategy
#     strategy = TrainingStrategy(
#         freeze_policy=BackboneFreezePolicy(),
#         augmentation=CutMixAugmentation()
#     )
#     strategy.apply_to_model(attention_model)
    
#     return attention_model

# # INTERVIEW QUESTIONS:
# """
# 1. When would you choose inheritance over composition?
# 2. How do you handle shared state between composed components?
# 3. What are the performance implications of composition vs inheritance?
# 4. How would you implement dependency injection in this system?
# 5. Design a plugin system that doesn't break existing functionality.
# """