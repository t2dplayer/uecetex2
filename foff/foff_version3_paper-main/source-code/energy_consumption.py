import numpy as np
from typing import Tuple, List
from config import SimulationConfig
from device_manager import Device
from task_manager import Task
from wireless_5g import Wireless5G, WirelessParameters

class EnergyCalculator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.mcs_table = self._create_mcs_table()

    def _create_mcs_table(self):
        return {
            -5: 0.01, 0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8,
            4: 1.0, 5: 1.5, 6: 2.0, 7: 2.5, 8: 3.0,
            9: 3.5, 10: 4.0, 15: 4.5, 20: 5.0, 25: 5.5, 30: 6.0
        }

    def calculate_transmission_energy(self, distance: float, data_size_mb: float) -> float:
        """
        Calcula a energia de transmissão usando o modelo de consumo de energia do 5G NR.
        
        Parâmetros:
            - distance (float): Distância entre o dispositivo e o destino (em metros).
            - data_size_mb (float): Tamanho dos dados a serem transmitidos (em Megabytes).
            
        Retorna:
            - Energia consumida na transmissão (Joules).
        """
        # Constantes do modelo de consumo de energia
        power_amplifier_efficiency = 0.38  # Eficiência do amplificador de potência (~38%)
        circuit_power_tx = 1.5  # Consumo de energia do circuito de transmissão em Watts
        # Tempo necessário para transmitir os dados       
        transmission_time = self.config.wireless.calculate_transmission_time(distance, data_size_mb)
        # Consumo de energia do amplificador de potência
        power_consumed = self.config.tx_power_dbm / (10 * power_amplifier_efficiency)        
        # Energia total de transmissão (Joules)
        transmission_energy = (power_consumed + circuit_power_tx) * transmission_time
        
        return transmission_energy


    def calculate_computation_energy(self, cycles: int, kappa: float) -> float:
        """Calculate computation energy consumption"""
        return (kappa * cycles) * 1e-9  # Convert nJ to J

    def calculate_transmission_energy_dynamic(self, distance, data_size_mb, direction='uplink'):
        power_amplifier_efficiency = np.random.normal(0.38, 0.02)  # eficiência variável
        circuit_power_tx = 1.5 if direction == 'uplink' else 1.2   # downlink geralmente menor potência
        transmission_time = self.config.wireless.calculate_transmission_time(distance, data_size_mb)
        
        # Adicionar possibilidade de retransmissões devido ao PER:
        packet_error_rate = np.random.uniform(0.01, 0.05)  # erro típico entre 1% e 5%
        num_retransmissions = np.random.geometric(1 - packet_error_rate)
        
        # Potência de transmissão adaptativa conforme distância e SNR:
        adaptive_tx_power = self.config.tx_power_dbm + np.random.uniform(-1, 1)  # ajustes de potência
        
        power_consumed = adaptive_tx_power / (10 * power_amplifier_efficiency)
        transmission_energy = (power_consumed + circuit_power_tx) * transmission_time * num_retransmissions
        
        return transmission_energy


    def calculate_computation_energy_stochastic(self, cycles, frequency_ghz, kappa_nj):
        load_factor = np.random.uniform(0.8, 1.2)  # variação da carga CPU (80-120%)
        frequency_variation = np.random.normal(frequency_ghz, 0.1)  # variação frequência CPU
        
        comp_energy = kappa_nj * cycles * (frequency_variation**2) * load_factor * 1e-9
        return comp_energy


    def total_energy(self, src_device: Device, dst_device: Device, task:Task) -> float:
        """Calculate total energy for a task"""
        distance = np.linalg.norm(dst_device.position - src_device.position)
        
        # Energia de transmissão com variação dinâmica de potência e PER
        tx_energy_upload = self.calculate_transmission_energy_dynamic(distance, task.data_size, direction='uplink')
        tx_energy_download = self.calculate_transmission_energy_dynamic(distance, task.data_size, direction='downlink')
        
        # Energia computacional com variação de carga da CPU
        comp_energy = self.calculate_computation_energy_stochastic(
            task.cpu_cycles * task.data_size, 
            dst_device.frequency_ghz, 
            dst_device.kappa_nj
        )
        return tx_energy_upload + tx_energy_download + comp_energy