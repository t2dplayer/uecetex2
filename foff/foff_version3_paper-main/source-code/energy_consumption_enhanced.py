import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Literal
import matplotlib.pyplot as plt

class EnhancedEnergyCalculator:
    """
    Enhanced energy calculator for 5G devices that incorporates findings from research
    on 5G NR and RedCap device power consumption characteristics.
    """
    def __init__(self, device_type: Literal["NR", "RedCap"] = "NR", frequency_band: Literal["n41", "n78"] = "n41"):
        """
        Initialize the energy calculator with device type and frequency band.
        
        Parameters:
            device_type: Type of 5G device - "NR" (New Radio) or "RedCap" (Reduced Capability)
            frequency_band: Frequency band - "n41" (2500 MHz) or "n78" (3700 MHz)
        """
        self.device_type = device_type
        self.frequency_band = frequency_band
        
        # Power model parameters based on the research paper
        self.power_params = {
            "RedCap": {
                "n41": {"alpha1": 0.30, "beta1": 1.5e-4, "gamma1": 4, 
                       "alpha2": 0.19, "beta2": 2.1e-2, "pmax": 0.69},
                "n78": {"alpha1": 0.35, "beta1": 1.1e-4, "gamma1": 5, 
                       "alpha2": 0.46, "beta2": 3.6e-3, "pmax": 1.45}
            },
            "NR": {
                "n41": {"alpha1": 0.50, "beta1": 2.0e-4, "gamma1": 5, 
                       "alpha2": 0.30, "beta2": 3.0e-2, "pmax": 1.20},
                "n78": {"alpha1": 0.60, "beta1": 1.8e-4, "gamma1": 6, 
                       "alpha2": 0.50, "beta2": 4.0e-3, "pmax": 2.10}
            }
        }
        
        # Static power consumption in different device states (in mW)
        self.power_states = {
            "RedCap": {
                "idle": 21.9,                # eDRX in RRC Inactive
                "connected": 252.6,          # Connected but not transmitting
                "receiving": 366.3,          # Receiving data
                "transmitting": None,        # Calculated dynamically based on TX power
            },
            "NR": {
                "idle": 34.8,                # DRX
                "connected": 588.2,          # Connected but not transmitting
                "receiving": 1104.0,         # Receiving data
                "transmitting": None,        # Calculated dynamically based on TX power
            }
        }
        
        # Circuit parameters for transmission energy calculation
        self.tx_params = {
            "power_amplifier_efficiency": 0.38,  # Power amplifier efficiency (~38%)
            "circuit_power_tx": 1.5              # Circuit power consumption in Watts
        }
        
        # Create MCS table for data rate calculation
        self.mcs_table = self._create_mcs_table()
    
    def _create_mcs_table(self) -> Dict[int, float]:
        """Create a mapping of MCS values to data rates."""
        return {
            -5: 0.01, 0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8,
            4: 1.0, 5: 1.5, 6: 2.0, 7: 2.5, 8: 3.0,
            9: 3.5, 10: 4.0, 15: 4.5, 20: 5.0, 25: 5.5, 30: 6.0
        }
    
    def calculate_uplink_power_consumption(self, tx_power_dbm: float) -> float:
        """
        Calculate the device power consumption when transmitting with a given TX power.
        Implementation of the piecewise linear model from the research paper.
        
        Parameters:
            tx_power_dbm: Transmission power in dBm
            
        Returns:
            Power consumption in Watts
        """
        params = self.power_params[self.device_type][self.frequency_band]
        
        if tx_power_dbm >= params["gamma1"]:
            # Higher power region
            power = params["alpha2"] + params["beta2"] * tx_power_dbm
            # Cap at maximum power
            return min(power, params["pmax"])
        else:
            # Lower power region
            return params["alpha1"] + params["beta1"] * tx_power_dbm
    
    def calculate_transmission_energy(self, distance: float, data_size_mb: float, 
                                     tx_power_dbm: float, data_rate_mbps: float) -> float:
        """
        Calculate energy consumed for transmitting data.
        
        Parameters:
            distance: Distance between device and base station (meters)
            data_size_mb: Size of data to transmit (MB)
            tx_power_dbm: Transmission power (dBm)
            data_rate_mbps: Data rate (Mbps)
            
        Returns:
            Energy consumed (Joules)
        """
        # Convert dBm to linear power (Watts)
        tx_power_w = 10 ** ((tx_power_dbm - 30) / 10)
        
        # Calculate transmission time
        transmission_time = data_size_mb * 8 / data_rate_mbps  # in seconds
        
        # Calculate power consumption (simplified from the original function)
        power_consumed = tx_power_w / self.tx_params["power_amplifier_efficiency"]
        total_power = power_consumed + self.tx_params["circuit_power_tx"]
        
        # Calculate energy
        energy = total_power * transmission_time
        
        return energy
    
    def estimate_battery_life(self, battery_capacity_mah: int = 10000, battery_voltage: float = 3.7,
                            active_minutes_per_hour: float = 5, tx_power_dbm: float = 10,
                            data_size_mb_per_transmission: float = 5) -> float:
        """
        Estimate battery life in days based on usage pattern.
        
        Parameters:
            battery_capacity_mah: Battery capacity in mAh
            battery_voltage: Battery voltage in volts
            active_minutes_per_hour: Minutes of active transmission per hour
            tx_power_dbm: Transmission power in dBm
            data_size_mb_per_transmission: Data size transmitted each active period (MB)
            
        Returns:
            Estimated battery life in days
        """
        # Convert battery capacity to Watt-hours
        battery_capacity_wh = battery_capacity_mah * battery_voltage / 1000
        
        # Convert to Joules
        battery_energy_j = battery_capacity_wh * 3600
        
        # Calculate power consumption in different states
        tx_power_w = self.calculate_uplink_power_consumption(tx_power_dbm)
        
        # Time distribution (in hours per day)
        hours_active = active_minutes_per_hour / 60 * 24
        hours_idle = 24 - hours_active
        
        # Energy consumption per day
        energy_active_j = tx_power_w * hours_active * 3600  # Convert hours to seconds
        energy_idle_j = self.power_states[self.device_type]["idle"] / 1000 * hours_idle * 3600
        
        total_energy_per_day_j = energy_active_j + energy_idle_j
        
        # Calculate battery life in days
        battery_life_days = battery_energy_j / total_energy_per_day_j
        
        return battery_life_days
    
    def compare_energy_efficiency(self, snr_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare energy efficiency for NR and RedCap across different SNR values.
        
        Parameters:
            snr_values: Array of SNR values to evaluate
            
        Returns:
            Tuple containing arrays of power consumption for NR and RedCap
        """
        # Calculate TX power requirement based on SNR (simplified model)
        def estimate_tx_power(snr):
            # This is a simplified model - in reality, the relationship is more complex
            # At cell edge (low SNR), high TX power is needed
            if snr < 10:
                return 20 - snr * 0.5  # Higher power at cell edge
            else:
                # More variable in good conditions, but generally lower
                return max(0, 15 - snr * 0.4)
        
        # Calculate power consumption across SNR values
        original_device_type = self.device_type
        
        # Calculate for NR
        self.device_type = "NR"
        nr_power = np.array([self.calculate_uplink_power_consumption(estimate_tx_power(snr)) for snr in snr_values])
        
        # Calculate for RedCap
        self.device_type = "RedCap"
        redcap_power = np.array([self.calculate_uplink_power_consumption(estimate_tx_power(snr)) for snr in snr_values])
        
        # Restore original device type
        self.device_type = original_device_type
        
        return nr_power, redcap_power
    
    def plot_power_comparison(self) -> None:
        """
        Plot power consumption comparison between 5G NR and RedCap devices.
        """
        tx_power_values = np.linspace(-10, 20, 100)  # -10 to 20 dBm
        
        # Store current settings
        original_type = self.device_type
        original_band = self.frequency_band
        
        plt.figure(figsize=(10, 6))
        
        # Calculate for all combinations
        for device in ["NR", "RedCap"]:
            for band in ["n41", "n78"]:
                self.device_type = device
                self.frequency_band = band
                power_consumption = [self.calculate_uplink_power_consumption(p) for p in tx_power_values]
                plt.plot(tx_power_values, power_consumption, 
                         label=f"{device}, {band} ({2500 if band=='n41' else 3700} MHz)")
        
        # Restore original settings
        self.device_type = original_type
        self.frequency_band = original_band
        
        plt.xlabel('Uplink Transmit Power (dBm)')
        plt.ylabel('Power Consumption (W)')
        plt.title('5G Device Power Consumption vs. Transmit Power')
        plt.grid(True)
        plt.legend()
        plt.yscale('log')
        plt.axvspan(-10, 0, alpha=0.2, color='green', label='Cell Center')
        plt.axvspan(10, 20, alpha=0.2, color='red', label='Cell Edge')
        plt.tight_layout()
        
        plt.show()
    
    def calculate_pv_size_requirements(self, daily_data_mb: float, avg_snr: float,
                                     location_peak_sun_hours: float = 3.0) -> float:
        """
        Calculate the required photovoltaic cell size for continuous operation.
        
        Parameters:
            daily_data_mb: Daily data transmission requirement in MB
            avg_snr: Average signal-to-noise ratio at the location
            location_peak_sun_hours: Average peak sun hours per day at the location
            
        Returns:
            Required PV cell size in cm²
        """
        # Estimate TX power based on SNR
        if avg_snr < 10:
            tx_power_dbm = 20 - avg_snr * 0.5  # Higher power at cell edge
        else:
            tx_power_dbm = max(0, 15 - avg_snr * 0.4)
        
        # Calculate daily energy requirement
        power_w = self.calculate_uplink_power_consumption(tx_power_dbm)
        
        # Assume 5 minutes active transmission per hour
        active_hours = 5/60 * 24
        idle_hours = 24 - active_hours
        
        # Calculate daily energy requirement (Wh)
        energy_active_wh = power_w * active_hours
        energy_idle_wh = self.power_states[self.device_type]["idle"] / 1000 * idle_hours
        daily_energy_wh = energy_active_wh + energy_idle_wh
        
        # Calculate required PV capacity (assuming typical efficiency)
        pv_watt_per_cm2 = 0.02  # 200W/m² = 0.02W/cm²
        required_pv_size_cm2 = daily_energy_wh / (location_peak_sun_hours * pv_watt_per_cm2)
        
        return required_pv_size_cm2


# Example usage
if __name__ == "__main__":
    # Initialize the calculator for RedCap device in n41 band
    energy_calc = EnhancedEnergyCalculator(device_type="RedCap", frequency_band="n41")
    
    # Calculate power consumption at different TX power levels
    tx_power = 15  # dBm
    power = energy_calc.calculate_uplink_power_consumption(tx_power)
    print(f"Power consumption at {tx_power} dBm: {power:.3f} W")
    
    # Estimate battery life
    battery_life = energy_calc.estimate_battery_life(
        battery_capacity_mah=10000,
        active_minutes_per_hour=5,
        tx_power_dbm=tx_power
    )
    print(f"Estimated battery life: {battery_life:.1f} days")
    
    # Calculate required PV size
    pv_size = energy_calc.calculate_pv_size_requirements(
        daily_data_mb=100,
        avg_snr=20,
        location_peak_sun_hours=4.0
    )
    print(f"Required PV cell size: {pv_size:.1f} cm²")
    
    # Plot power comparison between device types and frequency bands
    energy_calc.plot_power_comparison()