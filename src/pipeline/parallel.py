"""
Multiprocessing Support for Experiment Pipeline

Implements parallel execution capabilities for:
- Running multiple techniques concurrently
- Evaluating multiple datasets in parallel
- Processing multiple samples simultaneously

Complies with Academic Submission Guidelines Chapter 16.
"""

from typing import List, Dict, Any, Callable, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import logging

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    Manages parallel execution of experiment tasks.

    Features:
    - Thread-safe operation
    - Automatic resource management
    - Error handling and recovery
    - Progress tracking
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum number of worker processes.
                        If None, uses cpu_count() - 1 to leave one core free.
        """
        if max_workers is None:
            # Leave one CPU free for system operations
            self.max_workers = max(1, cpu_count() - 1)
        else:
            self.max_workers = max(1, min(max_workers, cpu_count()))

        logger.info(f"ParallelExecutor initialized with {self.max_workers} workers")

    def map_parallel(
        self,
        func: Callable,
        items: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Apply function to items in parallel.

        Args:
            func: Function to apply to each item
            items: List of items to process
            **kwargs: Additional arguments to pass to func

        Returns:
            List of results in same order as input items

        Example:
            >>> executor = ParallelExecutor(max_workers=4)
            >>> results = executor.map_parallel(
            ...     process_sample,
            ...     samples,
            ...     model="gpt-4"
            ... )
        """
        if not items:
            return []

        # If only one item or one worker, run sequentially
        if len(items) == 1 or self.max_workers == 1:
            return [func(item, **kwargs) for item in items]

        # Create partial function with kwargs
        if kwargs:
            partial_func = partial(func, **kwargs)
        else:
            partial_func = func

        # Execute in parallel
        try:
            with Pool(processes=self.max_workers) as pool:
                results = pool.map(partial_func, items)
            return results
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fall back to sequential execution
            logger.warning("Falling back to sequential execution")
            return [func(item, **kwargs) for item in items]

    def map_parallel_with_progress(
        self,
        func: Callable,
        items: List[Any],
        desc: str = "Processing",
        **kwargs
    ) -> List[Any]:
        """
        Apply function to items in parallel with progress tracking.

        Args:
            func: Function to apply
            items: Items to process
            desc: Description for progress tracking
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        logger.info(f"{desc}: Processing {len(items)} items with {self.max_workers} workers")

        results = self.map_parallel(func, items, **kwargs)

        logger.info(f"{desc}: Completed {len(results)} items")
        return results

    def execute_parallel_tasks(
        self,
        tasks: List[Callable],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Execute multiple different tasks in parallel.

        Args:
            tasks: List of callable tasks to execute
            timeout: Optional timeout in seconds for each task

        Returns:
            List of results from each task

        Example:
            >>> tasks = [
            ...     lambda: evaluate_technique("cot"),
            ...     lambda: evaluate_technique("react"),
            ...     lambda: evaluate_technique("tot")
            ... ]
            >>> results = executor.execute_parallel_tasks(tasks)
        """
        if not tasks:
            return []

        if len(tasks) == 1 or self.max_workers == 1:
            return [task() for task in tasks]

        try:
            with Pool(processes=min(len(tasks), self.max_workers)) as pool:
                if timeout:
                    # Use async with timeout
                    async_results = [pool.apply_async(task) for task in tasks]
                    results = [ar.get(timeout=timeout) for ar in async_results]
                else:
                    results = pool.map(lambda task: task(), tasks)
            return results
        except Exception as e:
            logger.error(f"Parallel task execution failed: {e}")
            # Fall back to sequential
            return [task() for task in tasks]

    @staticmethod
    def get_optimal_workers(
        num_tasks: int,
        min_workers: int = 1,
        max_workers: Optional[int] = None
    ) -> int:
        """
        Calculate optimal number of workers for given tasks.

        Args:
            num_tasks: Number of tasks to process
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (None = cpu_count())

        Returns:
            Optimal number of workers
        """
        if max_workers is None:
            max_workers = cpu_count()

        # Don't use more workers than tasks
        # And leave one CPU free
        optimal = min(num_tasks, max(1, cpu_count() - 1))
        return max(min_workers, min(optimal, max_workers))


def parallel_evaluate_samples(
    evaluate_func: Callable,
    samples: List[Dict[str, Any]],
    max_workers: Optional[int] = None,
    **eval_kwargs
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple samples in parallel.

    Args:
        evaluate_func: Function to evaluate each sample
        samples: List of samples to evaluate
        max_workers: Maximum parallel workers
        **eval_kwargs: Additional evaluation arguments

    Returns:
        List of evaluation results

    Example:
        >>> from pipeline.parallel import parallel_evaluate_samples
        >>> results = parallel_evaluate_samples(
        ...     evaluate_sample,
        ...     dataset_samples,
        ...     max_workers=4,
        ...     model="gpt-4"
        ... )
    """
    executor = ParallelExecutor(max_workers=max_workers)
    return executor.map_parallel_with_progress(
        evaluate_func,
        samples,
        desc="Evaluating samples",
        **eval_kwargs
    )


def parallel_evaluate_techniques(
    evaluate_func: Callable,
    techniques: List[str],
    max_workers: Optional[int] = None,
    **eval_kwargs
) -> Dict[str, Any]:
    """
    Evaluate multiple techniques in parallel.

    Args:
        evaluate_func: Function to evaluate each technique
        techniques: List of technique names
        max_workers: Maximum parallel workers
        **eval_kwargs: Additional evaluation arguments

    Returns:
        Dictionary mapping technique names to results

    Example:
        >>> from pipeline.parallel import parallel_evaluate_techniques
        >>> results = parallel_evaluate_techniques(
        ...     evaluate_technique,
        ...     ["baseline", "cot", "react"],
        ...     max_workers=3,
        ...     dataset=dataset
        ... )
    """
    executor = ParallelExecutor(max_workers=max_workers)

    # Create evaluation tasks
    tasks = [
        partial(evaluate_func, technique, **eval_kwargs)
        for technique in techniques
    ]

    # Execute in parallel
    results_list = executor.execute_parallel_tasks(tasks)

    # Map results back to technique names
    return dict(zip(techniques, results_list))


__all__ = [
    "ParallelExecutor",
    "parallel_evaluate_samples",
    "parallel_evaluate_techniques"
]
