from functional.transformations import CACHE_T
from functional.transformations import EXECUTION_STRATEGIES


class Lineage(object):
    def __init__(self, prior_lineage=None):
        self.transformations = [] if prior_lineage is None else list(prior_lineage.transformations)

    def __repr__(self):
        return 'Lineage: ' + ' -> '.join(
            ['sequence'] + [transform.name for transform in self.transformations]
        )

    def __len__(self):
        return len(self.transformations)

    def __getitem__(self, item):
        return self.transformations[item]

    def apply(self, transform):
        self.transformations.append(transform)

    def evaluate(self, sequence):
        result = sequence
        last_cache_index = self.cache_scan()
        for transform in self.transformations[last_cache_index:]:
            if transform.execution_strategies is not None \
                    and EXECUTION_STRATEGIES.PRE_COMPUTE in transform.execution_strategies:
                result = transform.function(list(result))
            else:
                result = transform.function(result)
        return iter(result)

    def cache_scan(self):
        try:
            return len(self.transformations) - self.transformations[::-1].index(CACHE_T)
        except ValueError:
            return 0
