import random
from offloading_strategy import OffloadingStrategy
from device_manager import Device, DeviceManager
import logging

class GTTStrategy(OffloadingStrategy):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.logger = logging.getLogger(__name__)
        self.counter = 0
    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager) -> Device:
        best_device = device_manager.get_device(0)
        best_latency, _ = self.simulator._calculate_task_metrics(tasks[current_task_index], best_device)
        for device in device_manager.devices:
            # if device.device_id == 1000 and self.counter % 500 == 0:                
            #     continue
            total_latency, _ = self.simulator._calculate_task_metrics(tasks[current_task_index], device)
            if total_latency < best_latency:
                best_device = device
                best_latency = total_latency
        self.logger.debug(f"GTT: Task {current_task_index} scheduled to device {best_device.device_id}")
        if best_device.device_id == 1000:
            self.counter += 1
        return best_device, None
        # return device_manager.get_device(1000), None
        
    def __str__(self):
        return 'GTT'
    
    def __repr__(self):
        return self.__str__()
