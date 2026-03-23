import random
from offloading_strategy import OffloadingStrategy
from device_manager import Device, DeviceManager
import logging

class RandomStrategy(OffloadingStrategy):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.logger = logging.getLogger(__name__)    
    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager) -> Device:
        """Random selection implementation"""
        best_device = random.choice(device_manager.devices)
        self.logger.debug(f"ROFF: Task {current_task_index} scheduled to device {best_device.device_id}")
        return best_device, None
    
    def __str__(self):
        return 'ROFF'
    
    def __repr__(self):
        return self.__str__()
