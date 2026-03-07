from typing import List, Dict
import numpy as np

class SimulationResult:
    def __init__(self, offloading_strategy: str):
        self.offloading_strategy = offloading_strategy
        self.total_latency: float = 0
        self.total_energy: float = 0
        self.processed_cycles: float = 0
        self.device_selections: List[int] = []
        self.ranking = []
        self.total_misses = 0
        self.invalid_arrivals = 0

    def increment_misses(self):
        self.total_misses += 1

    def increment_invalid_arrivals(self):
        self.invalid_arrivals += 1

    def add_result(self, latency: float, energy: float, cycles: int, device_id: int):
        """Adiciona um novo resultado ao conjunto de simulação."""
        self.total_latency += latency
        self.total_energy += energy
        self.processed_cycles += cycles
        self.device_selections.append(device_id)

    def add_ranking(self, ranking):
        self.ranking.append(ranking)

    def __str__(self):
        result  = f"Total Latency: {self.total_latency} second(s)\n"
        result += f"Total Energy: {self.total_energy} Joule(s)\n"
        result += f"Total Processed Cycles: {self.processed_cycles} cycles\n"
        return result

    def __repr__(self):
        arr = np.array([self.total_latency, self.total_energy, self.processed_cycles])
        return np.array2string(arr)

    def _compute_device_usage(self) -> Dict[int, float]:
        """Calcula a distribuição percentual de uso dos dispositivos."""
        if not self.device_selections:
            return {}
        unique, counts = np.unique(self.device_selections, return_counts=True)
        total = sum(counts)
        return {int(device_id): (count / total) * 100 for device_id, count in zip(unique, counts)}