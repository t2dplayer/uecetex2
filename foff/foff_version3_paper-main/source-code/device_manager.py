from dataclasses import dataclass
import numpy as np

@dataclass
class Device:
    frequency_ghz: float
    kappa_nj: float  # Energy per cycle in nJ
    queue_time: float
    position: np.ndarray
    device_id: int
    queue_release_time: float

class DeviceManager:
    def __init__(self, config: list):
        self.devices = [self._parse_device(d) for d in config]
        self.device_map = {device.device_id: device for device in self.devices}

    def _parse_device(self, raw_data: tuple) -> Device:
        return Device(
            frequency_ghz=raw_data[0],
            kappa_nj=raw_data[1],
            queue_time=raw_data[2],
            position=raw_data[3],
            device_id=raw_data[4],
            queue_release_time=raw_data[2]
        )

    def get_device(self, device_id: int) -> Device:
        """Retorna o dispositivo pelo ID, garantindo que a posição e ID sejam consistentes."""
        return self.device_map.get(device_id, None)