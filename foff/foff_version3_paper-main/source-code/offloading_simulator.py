import numpy as np
import random
from config import SimulationConfig, configure_logging
from device_manager import DeviceManager
from task_manager import Task, TaskManager
from energy_consumption import EnergyCalculator
from wireless_5g import Wireless5G, WirelessParameters
from simulation_result import SimulationResult
from offloading_strategy import OffloadingStrategy
from device_manager import Device

class OffloadingSimulator:
    def __init__(self, config: SimulationConfig, obui_index: int):
        self.config = config
        self.logger = configure_logging()
        self.energy_calculator = EnergyCalculator(config)
        self.obui_index = obui_index

        wireless_params = WirelessParameters(
            frequency_ghz=config.frequency_ghz,
            bandwidth_hz=config.bandwidth_hz,
            tx_power_dbm=config.tx_power_dbm,
            antenna_height=config.antenna_height,
            los_type=config.los_type,
            noise_density_dbm_hz=config.noise_density_dbm_hz
        )
        
        self.wireless = Wireless5G(wireless_params)
    
    def set_devices(self, devices):
        self.device_manager = DeviceManager(devices)
        device = self.device_manager.get_device(self.obui_index)
        if device:
            self.obui_position = device.position

    def _generate_tasks(self, num_tasks):
        """Generate random tasks with uniform distribution"""
        # random.seed(self.config.seed)
        return np.array([
            TaskManager._parse_task([
                np.random.randint(*self.config.data_size_range),
                np.random.randint(*self.config.cycles_per_mb_range),
                index,
                np.random.uniform(0, self.config.simulation_time * 0.5)
            ]) for index in range(num_tasks)
        ])

    def ghz_to_hz(self, frequency_ghz: float) -> float:
        return frequency_ghz * 1e9

    def _calculate_task_metrics(self, task: Task, device: Device):
        obui = self.device_manager.get_device(self.obui_index) # Obtém o dispositivo OBUI
        # Calcula o tempo de upload
        upload_distance = np.linalg.norm(obui.position - device.position)
        upload_time = self.wireless.calculate_transmission_time(upload_distance, task.data_size)
        # Calcula o tempo de download
        download_distance = np.linalg.norm(device.position - obui.position)
        download_time = self.wireless.calculate_transmission_time(download_distance, task.data_size)
        # Calcula tempo de processamento
        processing_time = (task.data_size * task.cpu_cycles) / self.ghz_to_hz(device.frequency_ghz)
        # Tempo total da tarefa
        total_latency = upload_time + download_time + device.queue_time + processing_time
        # Calcula a energia total
        total_energy = self.energy_calculator.total_energy(obui, device, task)
        return total_latency, total_energy 
    
    def release_device_queue(self, current_time: float, device_manager: DeviceManager):
        for device in device_manager.devices:
            if device.queue_release_time <= current_time:
                device.queue_time = 0.0

    def run_simulation(
            self, 
            tasks, 
            offloading_strategy: OffloadingStrategy, 
            sensitivity_test: bool = False
        ) -> SimulationResult:
        """Execute simulation with specified strategy"""
        result = SimulationResult(offloading_strategy.__str__())
        current_time = 0.0
        sorted_tasks = sorted(tasks, key=lambda x: x.arrival_time)
        # print(offloading_strategy.__str__())
        for task in sorted_tasks:
            if task.arrival_time > self.config.simulation_time:
                result.increment_invalid_arrivals()
                break  # Ignora tarefas fora do tempo de simulação 
            self.release_device_queue(current_time, self.device_manager)
            selected_device, ranking = offloading_strategy.execute(tasks, task.task_id, self.device_manager)
            if selected_device is None:
                selected_device = self.device_manager.get_device(self.obui_index)
            start_time = max(task.arrival_time, current_time + selected_device.queue_time)
            total_latency, total_energy = self._calculate_task_metrics(task, selected_device)            
            end_time = start_time + total_latency
            if end_time > self.config.simulation_time:
                result.increment_misses()                
                # continue
            if sensitivity_test == True:
                result.add_ranking(ranking)
            result.add_result(total_latency, total_energy, task.data_size * task.cpu_cycles, selected_device.device_id)
            selected_device.queue_release_time = end_time
            selected_device.queue_time = max(0, end_time - current_time)
            current_time = max(current_time, end_time)  # Avança o tempo global
            # if selected_device.device_id == 5:
            #     print(f"Release time: {selected_device.queue_release_time}")

        self.release_device_queue(current_time, self.device_manager)
        return result
