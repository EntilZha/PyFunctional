from functional.util import compose, parallelize


class ExecutionStrategies(object):
    """
    Enum like object listing the types of execution strategies
    """
    PRE_COMPUTE = 0
    PARALLEL = 1


class ExecutionEngine(object):

    def evaluate(self, sequence, transformations):
        result = sequence
        for transform in transformations:
            strategies = transform.execution_strategies
            if (strategies is not None
                    and ExecutionStrategies.PRE_COMPUTE in strategies):
                result = transform.function(list(result))
            else:
                result = transform.function(result)
        return iter(result)


class ParallelExecutionEngine(ExecutionEngine):

    def evaluate(self, sequence, transformations):
        result = sequence
        staged = []
        for transform in transformations:
            strategies = transform.execution_strategies
            if strategies and ExecutionStrategies.PRE_COMPUTE in strategies:
                result = list(result)
            if strategies and ExecutionStrategies.PARALLEL in strategies:
                staged.insert(0, transform.function)
            else:
                if staged:
                    result = parallelize(compose(*staged), result)
                    staged = []
                result = transform.function(result)
        if staged:
            result = parallelize(compose(*staged), result)
        return iter(result)
