"""
CHALLENGE: You're given a slow data preprocessing pipeline. 
Make it production-ready for a system processing 1M+ samples/day.
"""

import numpy as np
import time
from typing import Iterator, List, Tuple, Generator
from functools import reduce
from itertools import islice
import heapq

# SLOW IMPLEMENTATION (typical junior engineer code):
def slow_preprocess(dataset: List[dict]) -> List[Tuple[np.ndarray, int]]:
    """
    This processes 1000 samples/second. We need 10x faster.
    """
    result = []
    for item in dataset:
        # Load and process image
        image = np.array(item['image_data'])  # Assume loaded
        image = image.astype(np.float32) / 255.0
        
        # Normalize per-channel
        for channel in range(3):
            mean = np.mean(image[:, :, channel])
            std = np.std(image[:, :, channel])
            image[:, :, channel] = (image[:, :, channel] - mean) / std
        
        # Data augmentation
        if item['augment']:
            image = np.flip(image, axis=1)  # Horizontal flip
            
        # Convert label
        label = item['label']
        
        result.append((image, label))
    
    return result

# YOUR TASK: Rewrite for 10x performance
def fast_preprocess(dataset: List[dict]) -> Iterator[Tuple[np.ndarray, int]]:
    """
    TODO: Optimize using:
    1. Vectorization instead of loops
    2. Generator expressions for memory efficiency
    3. List comprehensions where appropriate
    4. NumPy broadcasting
    5. Eliminate unnecessary copies
    
    Target: Process 10,000+ samples/second
    """
    pass

# ADVANCED CHALLENGE: Distributed Serving
class DistributedInferenceCluster:
    """
    TODO: Coordinate inference across multiple nodes.
    
    Features:
    1. Automatic node discovery and health monitoring
    2. Load balancing with locality awareness
    3. Model replication and consistency
    4. Fault tolerance and automatic failover
    """
    
    def __init__(self, node_id: str, cluster_config: Dict[str, Any]):
        self.node_id = node_id
        self.cluster_config = cluster_config
        self.peer_nodes = {}  # Other nodes in cluster
        self.local_server = InferenceServer(cluster_config['local'])
        
        # Distributed coordination
        self.consensus_protocol = RaftConsensus(node_id)
        self.service_discovery = ServiceDiscovery()
        self.load_balancer = ConsistentHashBalancer()
    
    async def start_cluster_node(self):
        """TODO: Start node and join cluster"""
        # Start local server
        await self.local_server.start_server()
        
        # Join cluster
        await self.service_discovery.register_node(self.node_id)
        await self.discover_peers()
        
        # Start coordination tasks
        asyncio.create_task(self.monitor_cluster_health())
        asyncio.create_task(self.balance_load_across_cluster())
    
    async def route_request_globally(self, request: InferenceRequest) -> Dict[str, Any]:
        """
        TODO: Route request to best node in cluster.
        
        Consider:
        1. Node load and capacity
        2. Network latency to requester
        3. Model availability per node
        4. Data locality if applicable
        """
        pass
    
    async def replicate_model(self, model_name: str, target_nodes: List[str]):
        """TODO: Replicate model to specified nodes for availability"""
        pass

# PERFORMANCE TESTING FRAMEWORK
class LoadTester:
    """
    TODO: Comprehensive load testing for inference system.
    
    Test scenarios:
    1. Sustained high load (10K QPS for hours)
    2. Burst traffic patterns
    3. Mixed model workloads
    4. Node failure scenarios
    5. Model update during traffic
    """
    
    def __init__(self, target_url: str):
        self.target_url = target_url
        self.session_pool = aiohttp.ClientSession()
        self.results = LoadTestResults()
    
    async def run_sustained_load_test(self, 
                                    qps: int,
                                    duration_seconds: int,
                                    model_mix: Dict[str, float]) -> Dict[str, Any]:
        """TODO: Run sustained load test with specified QPS"""
        pass
    
    async def run_burst_test(self,
                           normal_qps: int,
                           burst_qps: int, 
                           burst_duration: int) -> Dict[str, Any]:
        """TODO: Test system behavior under traffic bursts"""
        pass
    
    async def chaos_test(self, chaos_scenarios: List[str]) -> Dict[str, Any]:
        """
        TODO: Chaos engineering tests.
        
        Scenarios:
        - Random node failures
        - Network partitions
        - Memory pressure
        - Disk failures
        - Model corruption
        """
        pass

# MONITORING AND OBSERVABILITY
class MetricsCollector:
    """
    TODO: Comprehensive metrics collection for production monitoring.
    
    Metrics categories:
    1. Request metrics (latency, throughput, errors)
    2. System metrics (CPU, memory, GPU utilization)  
    3. Model metrics (accuracy drift, prediction confidence)
    4. Business metrics (revenue impact, user satisfaction)
    """
    
    def __init__(self):
        self.prometheus_client = PrometheusClient()
        self.custom_metrics = {}
        self.alert_thresholds = {}
    
    def record_request_metric(self, 
                            model_name: str,
                            latency: float,
                            success: bool,
                            batch_size: int = 1):
        """TODO: Record request-level metrics"""
        pass
    
    def record_system_metric(self, metric_name: str, value: float):
        """TODO: Record system-level metrics"""
        pass
    
    def check_alert_conditions(self) -> List[Dict[str, Any]]:
        """TODO: Check if any metrics exceed alert thresholds"""
        pass

# INTERVIEW QUESTIONS:
"""
1. How do you handle backpressure when request rate exceeds processing capacity?
2. What's the trade-off between latency and throughput in batching?
3. How would you implement gradual model rollout (canary deployment)?
4. Design a caching layer for frequently requested predictions.
5. How do you ensure consistent model versions across distributed nodes?
6. What monitoring metrics are most critical for production ML systems?
7. How would you implement request prioritization without starvation?
8. Design a system for A/B testing different model versions in production.
"""

# SYSTEM DESIGN CHALLENGE:
"""
Design a complete ML serving platform that handles:

1. **Scale Requirements:**
   - 100,000+ QPS across all models
   - Sub-100ms P95 latency
   - 99.99% uptime SLA
   - Global deployment (multi-region)

2. **Model Management:**
   - 1000+ different models
   - Frequent model updates (multiple per day)
   - A/B testing capabilities
   - Automatic rollback on degradation

3. **Resource Optimization:**
   - Dynamic auto-scaling based on demand
   - Multi-tenant GPU sharing
   - Cost optimization across cloud providers
   - Edge deployment for low latency

4. **Reliability:**
   - Fault tolerance (node, network, model failures)
   - Circuit breakers and graceful degradation
   - Data consistency in distributed setup
   - Disaster recovery procedures

5. **Observability:**
   - Real-time performance monitoring
   - Model accuracy tracking in production
   - Cost tracking and optimization
   - Security and compliance auditing

How would you architect this system? What technologies and patterns would you use?
"""


"""
### Chapter 17 - Iterators, Generators, and Classic Coroutines
**🎯 Big Tech Question**: "Our training data is 100TB. Design a memory-efficient processing system."

**Exercise 17.1: Large-Scale Data Processing**

CHALLENGE: Process datasets that don't fit in memory efficiently.
Real scenario: Training foundation models on web-scale data.
"""

import itertools
from typing import Iterator, Generator, AsyncIterator, Tuple, Dict, Any, Optional
import asyncio
import aiofiles
import json
from pathlib import Path
import heapq
import random
import hashlib

class StreamingDataset:
    """
    TODO: Dataset that processes infinite streams of data.
    
    Requirements:
    1. Memory usage stays constant regardless of dataset size
    2. Support multiple data sources (files, databases, APIs)
    3. Handle data corruption gracefully
    4. Deterministic shuffling for reproducibility
    5. Efficient filtering and transformation
    """
    
    def __init__(self, 
                 sources: List[str],
                 batch_size: int = 32,
                 shuffle_buffer_size: int = 10000,
                 num_workers: int = 4):
        self.sources = sources
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self._transforms = []
        self._filters = []
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        TODO: Return iterator over all data sources.
        
        Should:
        1. Interleave multiple sources fairly
        2. Apply transformations lazily
        3. Handle source failures gracefully
        4. Maintain deterministic order when needed
        """
        pass
    
    def shuffle(self, buffer_size: Optional[int] = None) -> 'StreamingDataset':
        """
        TODO: Return shuffled view using reservoir sampling.
        
        Algorithm: Maintain sliding window buffer, randomly replace items
        Memory usage: O(buffer_size) regardless of total dataset size
        """
        buffer_size = buffer_size or self.shuffle_buffer_size
        
        def shuffled_iterator():
            buffer = []
            for item in self:
                if len(buffer) < buffer_size:
                    buffer.append(item)
                else:
                    # Reservoir sampling: randomly replace item in buffer
                    idx = random.randint(0, len(buffer))
                    if idx < len(buffer):
                        yield buffer[idx]
                        buffer[idx] = item
                    else:
                        yield item
            
            # Drain remaining buffer in random order
            random.shuffle(buffer)
            yield from buffer
        
        # TODO: Wrap in new StreamingDataset instance
        pass
    
    def filter(self, predicate) -> 'StreamingDataset':
        """TODO: Return filtered view"""
        pass
    
    def map(self, transform_fn) -> 'StreamingDataset':
        """TODO: Return transformed view"""
        pass
    
    def batch(self, batch_size: Optional[int] = None) -> Iterator[List[Dict[str, Any]]]:
        """
        TODO: Yield batches of specified size.
        
        Handle:
        1. Last batch may be smaller
        2. Dynamic batching based on memory usage
        3. Padding for models that require fixed batch sizes
        """
        pass
    
    def take(self, n: int) -> Iterator[Dict[str, Any]]:
        """TODO: Take first n items (for debugging/testing)"""
        pass
    
    def skip(self, n: int) -> 'StreamingDataset':
        """TODO: Skip first n items"""
        pass

async def async_data_loader(file_paths: List[str]) -> AsyncIterator[Dict[str, Any]]:
    """
    TODO: Asynchronously load data from multiple files.
    
    Features:
    1. Concurrent file reading
    2. Backpressure handling
    3. Error recovery and retry
    4. Progress tracking
    """
    semaphore = asyncio.Semaphore(10)  # Limit concurrent files
    
    async def load_single_file(path: str) -> AsyncIterator[Dict[str, Any]]:
        async with semaphore:
            # TODO: Implement async file reading with error handling
            pass
    
    # TODO: Merge multiple async iterators fairly
    pass

def external_sort(data_iterator: Iterator[Any], 
                 key_func: callable,
                 chunk_size: int = 10000,
                 temp_dir: str = '/tmp') -> Iterator[Any]:
    """
    TODO: Sort dataset larger than memory using external sorting.
    
    Algorithm:
    1. Read chunks of data, sort in memory, write to temp files
    2. Merge sorted temp files using heap
    3. Clean up temp files
    
    Used for: Sorting training data by sequence length for efficiency
    """
    temp_files = []
    chunk_num = 0
    
    # Phase 1: Create sorted chunks
    while True:
        chunk = list(itertools.islice(data_iterator, chunk_size))
        if not chunk:
            break
            
        # Sort chunk in memory
        chunk.sort(key=key_func)
        
        # Write to temp file
        temp_file = Path(temp_dir) / f'chunk_{chunk_num}.tmp'
        # TODO: Write chunk to file
        temp_files.append(temp_file)
        chunk_num += 1
    
    # Phase 2: Merge sorted files
    def merge_sorted_files():
        file_iterators = []
        # TODO: Open all temp files and create heap for merging
        pass
    
    try:
        yield from merge_sorted_files()
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)

class DataPipeline:
    """
    TODO: Composable data processing pipeline using generators.
    
    Example:
    pipeline = (DataPipeline(sources)
                .shuffle(10000)
                .filter(is_valid)
                .map(preprocess)
                .batch(32)
                .prefetch(2))
    
    for batch in pipeline:
        train_step(batch)
    """
    
    def __init__(self, data_source: Iterator):
        self._source = data_source
        self._operations = []
    
    def shuffle(self, buffer_size: int) -> 'DataPipeline':
        """Add shuffle operation to pipeline"""
        # TODO: Add shuffle operation without executing yet (lazy)
        pass
    
    def filter(self, predicate) -> 'DataPipeline':
        """Add filter operation"""
        pass
    
    def map(self, transform_fn, num_workers: int = 1) -> 'DataPipeline':
        """Add map operation, optionally with multiprocessing"""
        pass
    
    def batch(self, batch_size: int) -> 'DataPipeline':
        """Add batching operation"""
        pass
    
    def prefetch(self, buffer_size: int) -> 'DataPipeline':
        """Add prefetching with background thread"""
        pass
    
    def __iter__(self):
        """Execute pipeline lazily"""
        # TODO: Chain all operations using generators
        pass

# ADVANCED: Coroutine-based data processing
def coroutine_processor():
    """
    TODO: Use coroutines for push-based data processing.
    
    Benefits over pull-based iterators:
    1. Better for streaming/real-time data
    2. More efficient memory usage
    3. Easier backpressure handling
    """
    
    @coroutine
    def filter_stage(predicate, target):
        """Filter stage that forwards matching items"""
        while True:
            item = (yield)
            if predicate(item):
                target.send(item)
    
    @coroutine  
    def transform_stage(transform_fn, target):
        """Transform stage"""
        while True:
            item = (yield)
            transformed = transform_fn(item)
            target.send(transformed)
    
    @coroutine
    def batch_stage(batch_size, target):
        """Batching stage"""
        batch = []
        while True:
            item = (yield)
            batch.append(item)
            if len(batch) >= batch_size:
                target.send(batch)
                batch = []
    
    # TODO: Build processing pipeline using coroutines
    pass

def coroutine(func):
    """Decorator to prime coroutines"""
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)  # Prime the coroutine
        return gen
    return wrapper

# PERFORMANCE COMPARISON
def benchmark_data_processing():
    """
    TODO: Benchmark different data processing approaches.
    
    Compare:
    1. List-based processing (loads everything in memory)
    2. Iterator-based processing  
    3. Generator-based processing
    4. Async iterator processing
    5. Coroutine-based processing
    
    Measure: Memory usage, throughput, latency
    """
    import memory_profiler
    import time
    
    # TODO: Implement comprehensive benchmarks
    pass

# INTERVIEW QUESTIONS:
"""
1. When would you use generators vs async generators vs coroutines?
2. How do you handle backpressure in a data processing pipeline?
3. What's the memory complexity of different shuffling algorithms?
4. How would you implement deterministic shuffling for reproducible training?
5. Design a system for processing streaming data with exactly-once semantics.
6. How do you handle schema evolution in streaming datasets?
7. What are the trade-offs between pull-based vs push-based data processing?
8. How would you implement data lineage tracking in a pipeline?
"""

# REAL-WORLD SCENARIO:
"""
You're tasked with preprocessing the Common Crawl dataset (petabytes):

1. **Scale:** 
   - 300TB+ of web crawl data
   - Must process on cluster of 1000 machines
   - Each machine has 64GB RAM, 16 cores

2. **Requirements:**
   - Extract and clean text from HTML
   - Deduplicate content (near-duplicate detection)
   - Language detection and filtering  
   - Quality filtering (remove spam, adult content)
   - Tokenization for language models

3. **Constraints:**
   - Process within 24 hours
   - Output must be deterministically ordered
   - Handle machine failures gracefully
   - Cost-efficient (minimize cloud spend)

Design the data processing pipeline using Python generators and iterators.
How would you handle distributed coordination, fault tolerance, and monitoring?
"""

"""
## Integration Challenge: Build a Complete ML Platform

**Final Exercise: Production ML Platform**

ULTIMATE CHALLENGE: Build a complete ML platform integrating all concepts.

This is the kind of system you'd build at Google DeepMind, NVIDIA, or Microsoft.
Your solution will demonstrate mastery of all Fluent Python concepts.
"""

class MLPlatform:
    """
    TODO: Complete ML platform integrating all learned concepts.
    
    Components:
    1. Model registry with dynamic loading (Ch 22)
    2. Distributed training coordinator (Ch 9, 19-20)
    3. Data processing pipelines (Ch 17)
    4. Model serving infrastructure (Ch 19-20)
    5. Experiment tracking and management (Ch 11-12)
    6. Resource management and optimization (Ch 6)
    
    Architecture Requirements:
    - Support 1000+ concurrent experiments
    - Handle petabyte-scale datasets
    - Serve 100K+ QPS inference
    - Multi-tenant isolation
    - Global deployment
    - Cost optimization
    """
    
    def __init__(self, config: PlatformConfig):
        # Model management (Chapter 22 concepts)
        self.model_registry = SmartModelRegistry()
        
        # Training infrastructure (Chapter 9, 19-20 concepts)  
        self.training_coordinator = DistributedTrainingCoordinator()
        
        # Data processing (Chapter 17 concepts)
        self.data_engine = StreamingDataEngine()
        
        # Model serving (Chapter 19-20 concepts)
        self.inference_cluster = InferenceCluster()
        
        # Experiment management (Chapter 11-12 concepts)
        self.experiment_manager = ExperimentManager()
        
        # Resource optimization (Chapter 6 concepts)
        self.resource_manager = ResourceOptimizer()
    
    async def submit_training_job(self, 
                                job_spec: TrainingJobSpec) -> TrainingJob:
        """
        TODO: Submit distributed training job.
        
        Should handle:
        1. Resource allocation and optimization
        2. Data pipeline setup
        3. Model compilation and distribution
        4. Fault tolerance and checkpointing
        5. Real-time monitoring and alerting
        """
        pass
    
    async def deploy_model(self,
                         model_id: str,
                         deployment_spec: DeploymentSpec) -> ModelDeployment:
        """
        TODO: Deploy model for inference.
        
        Should handle:
        1. Model optimization for target hardware
        2. Load balancing and auto-scaling  
        3. A/B testing and gradual rollout
        4. Performance monitoring
        5. Automatic rollback on issues
        """
        pass
    
    def create_experiment(self,
                         experiment_config: ExperimentConfig) -> Experiment:
        """
        TODO: Create new experiment with full lifecycle management.
        
        Should provide:
        1. Isolated compute and storage resources
        2. Reproducible environment setup
        3. Hyperparameter optimization
        4. Results tracking and comparison
        5. Collaboration features
        """
        pass
    
    async def optimize_platform(self):
        """
        TODO: Continuously optimize platform performance.
        
        Optimization areas:
        1. Resource utilization and cost
        2. Model serving latency and throughput  
        3. Training job efficiency
        4. Data pipeline performance
        5. Energy consumption
        """
        pass

# SUCCESS CRITERIA:
"""
Your ML platform implementation will be evaluated on:

1. **Code Quality:**
   - Proper use of Python idioms from all chapters
   - Clean abstractions and interfaces
   - Comprehensive error handling
   - Performance-conscious design

2. **System Design:**
   - Scalability to enterprise requirements  
   - Fault tolerance and reliability
   - Security and compliance considerations
   - Monitoring and observability

3. **Technical Depth:**
   - Understanding of distributed systems concepts
   - Knowledge of ML infrastructure patterns
   - Performance optimization techniques
   - Resource management strategies

4. **Innovation:**
   - Novel solutions to common problems
   - Efficient algorithms and data structures  
   - Creative use of Python language features
   - Forward-thinking architecture decisions

This is the level of systems thinking and Python mastery expected at top-tier
ML engineering roles. Your implementation should demonstrate both deep
technical knowledge and practical engineering judgment.
"""
def streaming_preprocess(data_stream: Iterator[dict]) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Process infinite stream of data with constant memory usage.
    Handle backpressure gracefully.
    """
    pass

# PERFORMANCE ANALYSIS:
def benchmark_implementations():
    """
    TODO: Create realistic benchmark comparing:
    1. Memory usage over time
    2. Processing speed
    3. CPU utilization
    4. Cache efficiency
    
    Use memory_profiler and cProfile
    """
    pass

# INTERVIEW QUESTIONS TO ANSWER:
"""
1. What's the Big O complexity of each approach?
2. How does memory access pattern affect performance?
3. When would you use list comprehension vs generator expression?
4. How do you handle data that doesn't fit in memory?
5. What are the trade-offs between speed and memory usage?
"""