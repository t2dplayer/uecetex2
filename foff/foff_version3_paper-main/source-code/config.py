from dataclasses import dataclass
import logging
import os
from wireless_5g import *

@dataclass
class SimulationConfig:
    # Wireless Parameters
    bandwidth_hz: float = 400e6
    frequency_ghz: float = 3.5
    noise_density_dbm_hz: float = -174
    tx_power_dbm: float = 23.0    

    # Environment Parameters
    antenna_height: float = 30.0
    street_width: float = 20.0
    settlement_type: str = "urban"
    los_type: str = "los"
    
    # Simulation Parameters
    OBUI_INDEX = 1000
    seed: int = 1978
    num_tasks: int = 100
    total_tests: int = 400
    data_size_range: tuple = (1, 1024)  # MB
    cycles_per_mb_range: tuple = (1, 20)
    simulation_time: int = 1500 # segundos
    
    # Fuzzy System
    fuzzy_resolution: int = 1000
    weight_combinations: dict = None

    wireless = Wireless5G(WirelessParameters(frequency_ghz, bandwidth_hz, tx_power_dbm, antenna_height, noise_density_dbm_hz, los_type))

    def __post_init__(self):
        self.weight_combinations = self._load_weight_combinations()

    def get_linguistic_terms(self):
        return {
            i + 1: pair for i, pair in enumerate([
                ['vl', 'vl'],
                ['vl', 'l'],
                ['vl', 'ml'],
                ['vl', 'm'],
                ['vl', 'mh'],
                ['vl', 'h'],
                ['vl', 'vh'],
                ['l', 'vl'],
                ['l', 'l'],
                ['l', 'ml'],
                ['l', 'm'],
                ['l', 'mh'],
                ['l', 'h'],
                ['l', 'vh'],
                ['ml', 'vl'],
                ['ml', 'l'],
                ['ml', 'ml'],
                ['ml', 'm'],
                ['ml', 'mh'],
                ['ml', 'h'],
                ['ml', 'vh'],
                ['m', 'vl'],
                ['m', 'l'],
                ['m', 'ml'],
                ['m', 'm'],
                ['m', 'mh'],
                ['m', 'h'],
                ['m', 'vh'],
                ['mh', 'vl'],
                ['mh', 'l'],
                ['mh', 'ml'],
                ['mh', 'm'],
                ['mh', 'mh'],
                ['mh', 'h'],
                ['mh', 'vh'],
                ['h', 'vl'],
                ['h', 'l'],
                ['h', 'ml'],
                ['h', 'm'],
                ['h', 'mh'],
                ['h', 'h'],
                ['h', 'vh'],
                ['vh', 'vl'],
                ['vh', 'l'],
                ['vh', 'ml'],
                ['vh', 'm'],
                ['vh', 'mh'],
                ['vh', 'h'],
                ['vh', 'vh']
            ])
        }

    def _load_weight_combinations(self):
        return self.get_linguistic_terms()

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)