class Lineage(object):
    def __init__(self, prior_lineage=None):
        self.transformations = [] if prior_lineage is None else list(prior_lineage.transformations)

    def __repr__(self):
        return 'Lineage: ' + ' -> '.join(['sequence'] + [transform.name for transform in self.transformations])

    def apply(self, transform):
        self.transformations.append(transform)

    def evaluate(self, sequence):
        result = sequence
        for transform in self.transformations:
            result = transform.function(result)
        return iter(result)
