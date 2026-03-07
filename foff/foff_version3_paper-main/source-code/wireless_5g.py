import numpy as np
from dataclasses import dataclass
from scipy.stats import rice

@dataclass
class WirelessParameters:
    frequency_ghz: float
    bandwidth_hz: float
    tx_power_dbm: float
    antenna_height: float
    noise_density_dbm_hz: float
    los_type: str
    

class Wireless5G:
    def __init__(self, params: WirelessParameters):
        self.params = params

    def calculate_path_loss(self, distance: float) -> float:
        """Calculate path loss using 5G NR UMa model"""
        if distance <= 0.0:
            return 0.0
        if self.params.los_type == 'los':
            return 32.4 + 21.0 * np.log10(distance) + 20.0 * np.log10(self.params.frequency_ghz)
        else:
            return 35.3 * np.log10(distance) + 22.4 + 21.3 * np.log10(self.params.frequency_ghz)

    # def calculate_snr(self, distance: float) -> float:
    #     """Calculate Signal-to-Noise Ratio"""
    #     path_loss = self.calculate_path_loss(distance)
    #     rx_power = self.params.tx_power_dbm - path_loss
    #     noise_power = 10**((self.params.noise_density_dbm_hz - 30)/10) * self.params.bandwidth_hz
    #     return 10**((rx_power - 30)/10) / noise_power  # Convert dBm to Watts

    def calculate_snr(self, distance: float) -> float:
        """Calcula SNR considerando fading e interferência intercelular"""
        path_loss = self.calculate_path_loss(distance)
        rx_power = self.params.tx_power_dbm - path_loss

        # Adicionando Fading Rayleigh para não-los e Rician para los
        fading_factor = np.random.rayleigh() if self.params.los_type == 'nlos' else rice.rvs(1, scale=0.5)
        rx_power += 10 * np.log10(fading_factor)  # Adiciona efeito do fading

        # Considerando interferência de células vizinhas
        num_interfering_cells = 3  # Número arbitrário de células interferentes
        interference_power = sum([self.params.tx_power_dbm - self.calculate_path_loss(distance * np.random.uniform(0.8, 1.2)) for _ in range(num_interfering_cells)])
        total_noise = 10**((self.params.noise_density_dbm_hz - 30)/10) * self.params.bandwidth_hz + 10**(interference_power / 10)
        
        return 10**((rx_power - 30)/10) / total_noise


    def calculate_throughput(self, distance: float) -> float:
        """Calculate maximum achievable throughput"""
        snr = self.calculate_snr(distance)
        return self.params.bandwidth_hz * np.log2(1 + snr)
    
    def calculate_mcs_index(self, snr: float) -> int:
        """Determina o índice MCS baseado no SNR."""
        mcs_table = {
            -5: 0, 0: 1, 1: 2, 2: 3, 3: 4,
            4: 5, 5: 6, 6: 7, 7: 8, 8: 9,
            9: 10, 10: 11, 15: 12, 20: 13, 25: 14, 30: 15
        }
        
        # Seleciona o maior índice MCS possível baseado no SNR
        closest_snr = max([k for k in mcs_table.keys() if k <= snr], default=-5)
        return mcs_table[closest_snr]

    def calculate_transmission_time(self, distance: float, data_size_mb: float) -> float:
        if distance <= 0.0:
            return 0.0
        """Calcula o tempo de transmissão de uma tarefa usando 5G NR"""
        snr = self.calculate_snr(distance)
        mcs_index = self.calculate_mcs_index(snr)
        throughput = self.calculate_throughput(distance) # bps
        if throughput <= 0:
          return np.inf
        transmission_time = (data_size_mb * 8e6) / throughput # Segundos
        return transmission_time