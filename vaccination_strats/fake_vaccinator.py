"""Placeholder vaccinator that performs no vaccinations."""
from typing import List

from vaccination_strats.vaccinator import Vaccinator


class FakeVaccinator(Vaccinator):
    """No-op vaccinator that returns an empty vaccination list."""
    def get_vaccination_list(self, current_infected) -> List[int]:
        """Return an empty vaccination list regardless of state.

        Args:
            current_infected (Sequence[int]): Nodes currently infected.

        Returns:
            List[int]: Always empty.
        """
        return []