"""
CHALLENGE: Build a data pipeline using composition instead of inheritance.
"""

# BAD: Inheritance-based pipeline
class DataPipeline:
    def process(self, data):
        raise NotImplementedError

class ImagePipeline(DataPipeline):
    def process(self, data):
        # Image-specific processing
        pass

class AugmentedImagePipeline(ImagePipeline):
    def process(self, data):
        data = super().process(data)
        # Add augmentation
        return data

# Becomes unmaintainable with many pipeline types...

# GOOD: Composition-based pipeline
from typing import List, Callable
from functools import reduce

class ProcessingStep(ABC):
    """Single step in data processing pipeline"""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass
    
    def __or__(self, other: 'ProcessingStep') -> 'Pipeline':
        """Enable pipeline composition with | operator"""
        return Pipeline([self, other])

class Pipeline(ProcessingStep):
    """
    TODO: Composable pipeline that chains processing steps.
    
    Features:
    1. Parallel execution where possible
    2. Error handling and recovery  
    3. Progress tracking
    4. Conditional execution
    5. Caching of intermediate results
    """
    
    def __init__(self, steps: List[ProcessingStep]):
        self.steps = steps
    
    def __call__(self, data: Any) -> Any:
        """Execute pipeline steps in sequence"""
        # TODO: Implement with error handling
        pass
    
    def __or__(self, other: ProcessingStep) -> 'Pipeline':
        """Chain with another step or pipeline"""
        if isinstance(other, Pipeline):
            return Pipeline(self.steps + other.steps)
        return Pipeline(self.steps + [other])
    
    def parallel(self, max_workers: int = None) -> 'ParallelPipeline':
        """Convert to parallel execution where possible"""
        # TODO: Identify parallelizable steps
        pass
    
    def cache_intermediate(self, cache_dir: str) -> 'CachedPipeline':
        """Add caching to pipeline"""
        # TODO: Implement intelligent caching
        pass

class LoadImage(ProcessingStep):
    """TODO: Load image from file or URL"""
    pass

class ResizeImage(ProcessingStep):
    """TODO: Resize image to target dimensions"""
    pass

class NormalizeImage(ProcessingStep):
    """TODO: Normalize image values"""
    pass

class RandomAugmentation(ProcessingStep):
    """TODO: Apply random augmentations"""
    pass

# Usage with fluent interface:
"""
pipeline = (LoadImage() | 
           ResizeImage((224, 224)) | 
           RandomAugmentation(p=0.5) |
           NormalizeImage())

parallel_pipeline = pipeline.parallel(max_workers=4)
cached_pipeline = parallel_pipeline.cache_intermediate('./cache')
"""

# ADVANCED: Strategy Pattern for Different Processing Modes
class ProcessingMode(ABC):
    """Strategy for how pipeline executes"""
    
    @abstractmethod
    def execute(self, steps: List[ProcessingStep], data: Any) -> Any:
        pass

class SequentialMode(ProcessingMode):
    """Execute steps one after another"""
    pass

class ParallelMode(ProcessingMode):
    """Execute independent steps in parallel"""
    pass

class StreamingMode(ProcessingMode):
    """Process data in streaming fashion"""
    pass

class AdaptivePipeline(Pipeline):
    """
    TODO: Pipeline that adapts execution strategy based on:
    1. Data size and type
    2. Available system resources
    3. Processing step dependencies
    4. Performance requirements
    """
    
    def __init__(self, steps: List[ProcessingStep]):
        super().__init__(steps)
        self.mode_selector = ProcessingModeSelector()
    
    def __call__(self, data: Any) -> Any:
        mode = self.mode_selector.select_mode(self.steps, data)
        return mode.execute(self.steps, data)