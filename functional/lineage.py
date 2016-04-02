from __future__ import absolute_import

from functional.transformations import CACHE_T
from functional.transformations import ExecutionStrategies
from functional.util import parallelize


class Lineage(object):
    """
    Class for tracking the lineage of transformations, and applying them to a given sequence.
    """
    def __init__(self, prior_lineage=None):
        """
        Construct an empty lineage if prior_lineage is None or if its not use it as the list of
        current transformations

        :param prior_lineage: Lineage object to inherit
        :return: new Lineage object
        """
        self.transformations = [] if prior_lineage is None else list(prior_lineage.transformations)

    def __repr__(self):
        """
        Returns readable representation of Lineage

        :return: readable Lineage
        """
        return 'Lineage: ' + ' -> '.join(
            ['sequence'] + [transform.name for transform in self.transformations]
        )

    def __len__(self):
        """
        Number of transformations in lineage

        :return: number of transformations
        """
        return len(self.transformations)

    def __getitem__(self, item):
        return self.transformations[item]

    def apply(self, transform):
        self.transformations.append(transform)

    def evaluate(self, sequence):
        result = sequence
        last_cache_index = self.cache_scan()
        for transform in self.transformations[last_cache_index:]:
            strategies = transform.execution_strategies
            if strategies is not None:
                if ExecutionStrategies.PRE_COMPUTE in strategies:
                    result = list(result)
                if ExecutionStrategies.PARALLEL in strategies:
                    result = parallelize(transform.function, result)
                else:
                    result = transform.function(result)
            else:
                result = transform.function(result)
        return iter(result)

    def cache_scan(self):
        try:
            return len(self.transformations) - self.transformations[::-1].index(CACHE_T)
        except ValueError:
            return 0
