from typing import List

from vaccination_strats.vaccinator import Vaccinator


class FakeVaccinator(Vaccinator):
    def get_vaccination_list(self, current_infected) -> List[int]:
        return []