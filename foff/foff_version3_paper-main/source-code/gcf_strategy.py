import random
from offloading_strategy import OffloadingStrategy
from device_manager import Device, DeviceManager
import logging

class GCFStrategy(OffloadingStrategy):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.logger = logging.getLogger(__name__)
    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager) -> tuple:
        best_device = device_manager.get_device(0)
        best_frequency = device_manager.get_device(0).frequency_ghz
        for device in device_manager.devices:
            frequency = device.frequency_ghz
            if frequency > best_frequency:
                best_device = device
                best_frequency = frequency
        self.logger.debug(f"GCF: Task {current_task_index} scheduled to device {best_device.device_id}")
        return best_device, None
        
    def __str__(self):
        return 'GCF'
    
    def __repr__(self):
        return self.__str__()
