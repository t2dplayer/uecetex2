"""
FOFF Simulation Core
Paper Title: FOFF: An Energy-Efficient Task Offloading in VEC-Enabled Vehicular Networks Using Fuzzy
TOPSIS
Author: Antonio S. S. Vieira (sergio.vieira@ifce.edu.br) - 2025-01
"""

import logging
import random
import numpy as np
import copy
from offloading_simulator import *
from random_strategy import RandomStrategy
from gtt_strategy import GTTStrategy
from gcf_strategy import GCFStrategy
from foff_strategy import DFOFFStrategy, WFOFFStrategy
from partitioned_offloading_strategy import FPSched
from hoff_strategy import HybridStrategy
from psoff_strategy import PSOFFStrategy
from config import configure_logging
from plotting import *
from file_utils import *
from task_manager import parse_task_string
import hashlib
import argparse
import json 
from scipy import stats

def hash_from_array(arr):
    array_string = str(arr).encode('utf-8')
    return hashlib.md5(array_string).hexdigest()

parser = argparse.ArgumentParser(description='FOFF Simulator.')
parser.add_argument(
    '--queue-waiting-time',
    type=float,
    default=100.0,
    help='Tempo de espera na fila de processamento do OBUI (default: 100.0)'
)
parser.add_argument(
    '--scenario',
    type=str,
    choices=['challenging', 'balanced'],
    default='challenging',
    help='Cenário a ser utilizado: "challenging" ou "balanced" (default: challenging)'
)
parser.add_argument(
    '--extract-method',
    type=str,
    choices=['extract_nft_map', 'extract_nft_weighted'],
    default='extract_nft_weighted',
    help='Método de extração a ser utilizado (default: extract_nft_weighted)'
)

args = parser.parse_args()

# Atualiza o queue_waiting_time a partir do argumento
queue_waiting_time = args.queue_waiting_time

# Aqui os 6 dispositivos e suas caracteísticas
# Cada dispositivo possui:
#   Frequencia do cpu em GHZ (f)
#   Coeficiente de custo energético nj/ciclo (kappa)
#   Fila de espera segundos (Q)
#   Coordenadas (x, y) para cálculo de distância. Em metros. (position)
#   Identificador (id)
# (f, kappa, Q, pos, id) 
# queue_waiting_time = 100.0
CHALLEGING_SCENARIO = [
    (2, 90, 0.0, np.array([451.36, 741.45]), 0),
    (4,  670, 0.0, np.array([642.61, 697.24]), 1),
    (2, 5, 0.0, np.array([722.94, 603.27]), 2),
    (4, 345, 0.0, np.array([670.94, 318.01]), 3),    
    (2, 88, 0.0, np.array([256.37, 503.55]), 4),    
    (0.005, 1310, queue_waiting_time, np.array([500.0, 500.0]), SimulationConfig.OBUI_INDEX),
]
CHALLEGING_SCENARIO_HASH = hash_from_array(CHALLEGING_SCENARIO)

# (f, kappa, Q, pos, id) 
BALANCED_SCENARIO = [
    (2, 50, 0.0, np.array([451.36, 741.45]), 0),
    (3, 110, 0.0, np.array([256.37, 503.55]), 1),
    (2, 100, 0.0, np.array([722.94, 603.27]), 2),
    (4, 230, 0.0, np.array([670.94, 318.01]), 3),
    (2, 120, 0.0, np.array([642.61, 697.24]), 4),
    (0.005, 200, queue_waiting_time, np.array([500.0, 500.0]), SimulationConfig.OBUI_INDEX),    
]
BALANCED_SCENARIO_HASH = hash_from_array(BALANCED_SCENARIO)

if args.scenario == 'challenging':
    CURRENT_SCENARIO = CHALLEGING_SCENARIO
else:
    CURRENT_SCENARIO = BALANCED_SCENARIO

CURRENT_SCENARIO_HASH = hash_from_array(CURRENT_SCENARIO)

def set_results(result, value, qw_key, data, metric, extract_method):
    metric_map = data[metric]
    if qw_key not in metric_map:
        metric_map[qw_key] = dict()
    if result.offloading_strategy not in metric_map[qw_key]:
        metric_map[qw_key][result.offloading_strategy] = dict({'value': []})
    metric_map[qw_key][result.offloading_strategy]['value'].append(value)


# -------------------- Main Execution --------------------
if __name__ == "__main__":
    logger = configure_logging(level=logging.DEBUG)
    config = SimulationConfig()
    random.seed(config.seed)

    # OBUI está sempre na última posição da lista de dispositivos
    obui_index = SimulationConfig.OBUI_INDEX
    # config.num_tasks = 2
    # config.total_tests = 2
    sim = OffloadingSimulator(config, obui_index)    
    # original_tasks = list(sim._generate_tasks(config.num_tasks))
    # tasks_shuffled = [random.sample(original_tasks, len(original_tasks)) for _ in range(config.total_tests)]
    # # save("tasks_shuffled.csv", tasks_shuffled)
    # tasks_shuffled = load("tasks_shuffled.csv")
    # tasks_shuffled = load("adversarial_tasks_final.csv")
    tasks_shuffled = load("adversarial_tasks.csv")
    for i in range(len(tasks_shuffled)):
        for j in range(len(tasks_shuffled[0])):
            tasks_shuffled[i][j] = parse_task_string(tasks_shuffled[i][j])
        
    from foff_strategy import extract_nft_map, extract_nft_weighted
    current_omega = ['vl', 'vh']
    strategies = {
        'FOFF': (DFOFFStrategy(sim, omega=current_omega), []),
        # 'WFOFF': (WFOFFStrategy(sim, omega=current_omega), []),
        'ROFF': (RandomStrategy(sim), []),
        'GTT' : (GTTStrategy(sim), []),
        'FP-Sched' : (FPSched(sim), []),
        'GCF' : (GCFStrategy(sim), []),
        'HOFF' : (HybridStrategy(sim), []),
        'PSOFF' : (PSOFFStrategy(sim), []),
    }
    columns = list(strategies.keys())
    scenario_str = f"Challenging" if CURRENT_SCENARIO_HASH == CHALLEGING_SCENARIO_HASH else f"Balanced"
    # Run all simulations and collect results
    for key, (off_strategy, all_results) in strategies.items():       
        for counter, tasks in enumerate(tasks_shuffled):
            sim.set_devices(copy.deepcopy(CURRENT_SCENARIO))
            # logger.debug(f"===== Test {counter + 1} =====")
            print(f"({counter + 1})", end='')
            result = sim.run_simulation(tasks, off_strategy)
            all_results.append(result)

data_filename = format_filename(f"{scenario_str}_data_filename") + ".json"
data = load_data(data_filename)

for key, (_, all_results) in strategies.items():
    for result in all_results:
        qw_key = str(int(queue_waiting_time))
        set_results(result, result.total_energy, qw_key, data, 'energy_consumption', 'extract_nft_map')
        set_results(result, result.total_latency, qw_key, data, 'processed_time', 'extract_nft_map')
        set_results(result, result.processed_cycles, qw_key, data, 'processed_cycles', 'extract_nft_map')        
        # set_results(result, result.total_energy, qw_key, data, 'energy_consumption', 'extract_nft_map')
        # set_results(result, result.total_latency, qw_key, data, 'processed_time', 'extract_nft_map')

save_data(data, data_filename)

# Aggregate energy and latency data for all strategies
# total_energy_data = []
# total_latency_data = []
# total_cycles_data = []
# total_select_data = []
# total_misses_data = []
# total_invalid_arrivals_data = []
# for key, (_, all_results) in strategies.items():
#     results = all_results
#     total_energy_data.append([result.total_energy for result in results])
#     total_latency_data.append([result.total_latency for result in results])
#     total_cycles_data.append([result.processed_cycles for result in results])
#     total_select_data.append([result.device_selections for result in results])
#     total_misses_data.append([result.total_misses for result in results])
#     total_invalid_arrivals_data.append([result.invalid_arrivals for result in results])

# # Convert to numpy arrays
# total_energy_data = np.array(total_energy_data)
# total_latency_data = np.array(total_latency_data)
# total_cycles_data = np.array(total_cycles_data)
# total_misses_data = np.array(total_misses_data)
# total_invalid_arrivals_data = np.array(total_invalid_arrivals_data)
# # Plotting
# # scenario_str = f"Challenging - \n{current_extract_method[0]} - {current_omega}" if CURRENT_SCENARIO_HASH == CHALLEGING_SCENARIO_HASH else f"Balanced \n{current_extract_method[0]} - {current_omega}"
# plot_device_selection_count(total_select_data, columns, CURRENT_SCENARIO)
# plot_invalid_arrivals(total_invalid_arrivals_data, columns)
# plot_misses(total_misses_data, columns)
# plot_bar_with_ci(total_cycles_data, columns, 0.95,
#                  title=f"Total Processed Cycles x Algorithm (CI 95%) - {scenario_str} Scenario", 
#                  filename=f"Total Processed Cycles x Algorithm (CI 95%) - {scenario_str} - {current_extract_method[0]} - {queue_waiting_time}",
#                  ylabel="Total Processed Cycles")
# plot_bar_with_ci(total_energy_data, columns, confidence=0.95, 
#                  title=f"Total Energy Consumption x Algorithm (CI 95%) - {scenario_str} Scenario", 
#                  filename=f"Total Energy Consumption x Algorithm (CI 95%) - {scenario_str} - {current_extract_method[0]} - {queue_waiting_time}",                   
#                  ylabel="Total Energy Consumption (Joules)")
# plot_bar_with_ci(total_latency_data, columns, 0.95,
#                  title=f"Total Processed Time x Algorithm (CI 95%) - {scenario_str} Scenario", 
#                  filename=f"Total Processed Time x Algorithm (CI 95%) - {scenario_str} - {current_extract_method[0]} - {queue_waiting_time}",                   
#                  ylabel="Total Processed Time (Seconds)")
