from device_manager import Device, DeviceManager

class OffloadingStrategy:
    """Base class for offloading strategies"""
    def __init__(self, simulator):
        self.simulator = simulator

    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager) -> tuple:
        raise NotImplementedError
    
    def __str__(self):
        return 'None'
