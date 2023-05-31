from functools import partial
from functional.util import compose, parallelize


class ExecutionStrategies(object):
    """
    Enum like object listing the types of execution strategies.
    """

    PRE_COMPUTE = 0
    PARALLEL = 1


class ExecutionEngine(object):
    """
    Class to perform serial execution of a Sequence evaluation.
    """

    def evaluate(self, sequence, transformations):
        """
        Execute the sequence of transformations in serial
        :param sequence: Sequence to evaluation
        :param transformations: Transformations to apply
        :return: Resulting sequence or value
        """
        result = sequence
        for transform in transformations:
            strategies = transform.execution_strategies
            if strategies is not None and ExecutionStrategies.PRE_COMPUTE in strategies:
                result = transform.function(list(result))
            else:
                result = transform.function(result)
        return iter(result)


class ParallelExecutionEngine(ExecutionEngine):
    """
    Class to perform parallel execution of a Sequence evaluation.
    """

    def __init__(self, processes=None, partition_size=None):
        """
        Set the number of processes for parallel execution.
        :param processes: Number of parallel Processes
        """
        super(ParallelExecutionEngine, self).__init__()
        self.processes = processes
        self.partition_size = partition_size

    def evaluate(self, sequence, transformations):
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
            strategies = transform.execution_strategies or {}
            if ExecutionStrategies.PARALLEL in strategies:
                staged.insert(0, transform.function)
            else:
                if staged:
                    result = parallel(compose(*staged), result)
                    staged = []
                if ExecutionStrategies.PRE_COMPUTE in strategies:
                    result = list(result)
                result = transform.function(result)
        if staged:
            result = parallel(compose(*staged), result)
        return iter(result)
