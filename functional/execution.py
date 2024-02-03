from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Iterable, Iterator, Optional

from functional.util import compose, parallelize

if TYPE_CHECKING:
    from functional.transformations import Transformation


class ExecutionStrategies:
    """
    Enum like object listing the types of execution strategies.
    """

    PARALLEL = 1


class ExecutionEngine:
    """
    Class to perform serial execution of a Sequence evaluation.
    """

    def evaluate(
        self, sequence: Iterable, transformations: Iterable[Transformation]
    ) -> Iterator:
        """
        Execute the sequence of transformations in serial
        :param sequence: Sequence to evaluation
        :param transformations: Transformations to apply
        :return: Resulting sequence or value
        """
        result = sequence
        for transform in transformations:
            result = transform.function(result)
        return iter(result)


class ParallelExecutionEngine(ExecutionEngine):
    """
    Class to perform parallel execution of a Sequence evaluation.
    """

    def __init__(
        self, processes: Optional[int] = None, partition_size: Optional[int] = None
    ):
        """
        Set the number of processes for parallel execution.
        :param processes: Number of parallel Processes
        """
        super().__init__()
        self.processes = processes
        self.partition_size = partition_size

    def evaluate(
        self, sequence: Iterable, transformations: Iterable[Transformation]
    ) -> Iterator:
        """
        Execute the sequence of transformations in parallel
        :param sequence: Sequence to evaluation
        :param transformations: Transformations to apply
        :return: Resulting sequence or value
        """
        result = sequence
        parallel = partial(
            parallelize, processes=self.processes, partition_size=self.partition_size
        )
        staged = []
        for transform in transformations:
            strategies = transform.execution_strategies
            if ExecutionStrategies.PARALLEL in strategies:
                staged.insert(0, transform.function)
            else:
                if staged:
                    result = parallel(compose(*staged), result)
                    staged = []
                result = transform.function(result)
        if staged:
            result = parallel(compose(*staged), result)
        return iter(result)
