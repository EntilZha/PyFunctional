from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Optional

from functional.execution import ExecutionEngine
from functional.transformations import CACHE_T, Transformation


class Lineage:
    """
    Class for tracking the lineage of transformations, and applying them to a given sequence.
    """

    transformations: list[Transformation]
    engine: ExecutionEngine

    def __init__(
        self,
        prior_lineage: Optional[Lineage] = None,
        engine: Optional[ExecutionEngine] = None,
    ):
        """
        Construct an empty lineage if prior_lineage is None or if its not use it as the list of
        current transformations

        :param prior_lineage: Lineage object to inherit
        :return: new Lineage object
        """
        self.transformations = (
            [] if prior_lineage is None else prior_lineage.transformations.copy()
        )
        self.engine = (
            (engine or ExecutionEngine())
            if prior_lineage is None
            else prior_lineage.engine
        )

    def __repr__(self) -> str:
        """
        Returns readable representation of Lineage

        :return: readable Lineage
        """
        return "Lineage: " + " -> ".join(
            ["sequence"] + [transform.name for transform in self.transformations]
        )

    def __len__(self) -> int:
        """
        Number of transformations in lineage

        :return: number of transformations
        """
        return len(self.transformations)

    def __getitem__(self, item: int) -> Transformation:
        """
        Return specific transformation in lineage.
        :param item: Transformation to retrieve
        :return: Requested transformation
        """
        return self.transformations[item]

    def apply(self, transform: Transformation):
        """
        Add the transformation to the lineage
        :param transform: Transformation to apply
        """
        self.transformations.append(transform)

    def evaluate(self, sequence: Iterable) -> Iterator:
        """
        Compute the lineage on the sequence.

        :param sequence: Sequence to compute
        :return: Evaluated sequence
        """
        last_cache_index = self.cache_scan()
        transformations = self.transformations[last_cache_index:]
        return self.engine.evaluate(sequence, transformations)

    def cache_scan(self) -> int:
        """
        Scan the lineage for the index of the most recent cache.
        :return: Index of most recent cache
        """
        try:
            return len(self.transformations) - self.transformations[::-1].index(CACHE_T)
        except ValueError:
            return 0
