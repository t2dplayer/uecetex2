"""
FOFF Parameter Sensitivity Analysis

This script performs a sensitivity analysis on the Fuzzy Offloading Framework (FOFF)
to evaluate how different weight selections affect the performance metrics of energy 
efficiency and decision accuracy when using the Map extraction method.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import copy
from tqdm import tqdm
import time
import os
import json
from collections import defaultdict

# Import FOFF modules
from config import SimulationConfig, configure_logging
from offloading_simulator import OffloadingSimulator
from foff_strategy import DFOFFStrategy, extract_nft_map
from random_strategy import RandomStrategy
from gtt_strategy import GTTStrategy
from gcf_strategy import GCFStrategy
from file_utils import load
from task_manager import parse_task_string

# Custom file saving function to handle numpy arrays
def save_data(data, filename):
    """Save data to a JSON file with numpy array handling"""
    with open(filename, 'w') as f:
        json.dump(data, f, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj.__dict__ if hasattr(obj, '__dict__') else obj, indent=4)

class FOFFSensitivityAnalyzer:
    """Class for analyzing the sensitivity of FOFF to parameter changes"""
    
    def __init__(self, config):
        """Initialize the sensitivity analyzer"""
        self.config = config
        self.obui_index = SimulationConfig.OBUI_INDEX
        self.simulator = OffloadingSimulator(config, self.obui_index)
        self.logger = configure_logging(level=logging.INFO)
        
        # Importance levels for fuzzy weights
        self.importances = ["vl", "l", "ml", "m", "mh", "h", "vh"]
        
        # Results storage
        self.results = defaultdict(dict)
        
    def generate_omega_combinations(self):
        """Generate weight combinations for testing"""
        # Focus on key combinations likely to show significant differences
        combinations = []
        
        # Include extremes and middle values
        for w1 in ["vl", "m", "vh"]:
            for w2 in ["vl", "m", "vh"]:
                combinations.append([w1, w2])
        
        # Add additional interesting combinations
        additional = [
            ["l", "h"],
            ["h", "l"],
            ["ml", "mh"],
            ["mh", "ml"]
        ]
        
        combinations.extend(additional)
        return combinations
    
    def run_analysis(self, devices_config, tasks, num_tests=5):
        """Run the sensitivity analysis"""
        
        # Create output directory
        os.makedirs("sensitivity_results", exist_ok=True)
        
        # Generate omega combinations to test
        omega_combinations = self.generate_omega_combinations()
        
        self.logger.info(f"Running sensitivity analysis with {len(omega_combinations)} omega combinations")
        
        # Results structure
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Create comparison baseline with Random strategy
        random_strategy = RandomStrategy(self.simulator)
        random_results = self._run_strategy_tests(random_strategy, devices_config, tasks, num_tests)
        
        # Also test GTT and GCF strategies for comprehensive comparison
        gtt_strategy = GTTStrategy(self.simulator)
        gtt_results = self._run_strategy_tests(gtt_strategy, devices_config, tasks, num_tests)
        
        gcf_strategy = GCFStrategy(self.simulator)
        gcf_results = self._run_strategy_tests(gcf_strategy, devices_config, tasks, num_tests)
        
        # Store baseline results
        results["Baseline"]["Random"] = {
            "metrics": random_results,
            "strategy": "Random"
        }
        
        results["Baseline"]["GTT"] = {
            "metrics": gtt_results,
            "strategy": "GTT"
        }
        
        results["Baseline"]["GCF"] = {
            "metrics": gcf_results,
            "strategy": "GCF"
        }
        
        # Run tests for each omega combination with Map extraction method
        for omega in tqdm(omega_combinations, desc=f"Testing weight combinations"):
            # Create FOFF strategy with current omega
            foff_strategy = DFOFFStrategy(self.simulator, omega=omega)
            
            # Run tests for FOFF
            foff_results = self._run_strategy_tests(foff_strategy, devices_config, tasks, num_tests)
            
            # Calculate improvements over baseline strategies
            foff_improvements = self._calculate_improvements(foff_results, random_results)
            foff_vs_gtt = self._calculate_improvements(foff_results, gtt_results)
            foff_vs_gcf = self._calculate_improvements(foff_results, gcf_results)
            
            # Store results
            omega_key = f"{omega[0]}_{omega[1]}"
            results["FOFF"][f"{omega_key}"] = {
                "metrics": foff_results,
                "improvements": foff_improvements,
                "vs_gtt": foff_vs_gtt,
                "vs_gcf": foff_vs_gcf,
                "omega": omega,
                "strategy": "FOFF"
            }
        
        # Save raw results
        save_data(results, "sensitivity_results/raw_results.json")
        
        return results
    
    def _run_strategy_tests(self, strategy, devices_config, all_tasks, num_tests):
        """Run tests with a specific strategy and collect results"""
        
        total_energy = []
        total_latency = []
        device_selections = []
        decision_times = []
        
        # Run multiple tests for statistical validity
        for test in range(num_tests):
            # Use a subset of tasks for faster analysis
            task_subset = all_tasks[:min(20, len(all_tasks))]
            
            test_energy = []
            test_latency = []
            test_selections = []
            test_times = []
            
            for tasks in task_subset:
                # Reset simulator with fresh devices
                self.simulator.set_devices(copy.deepcopy(devices_config))
                
                # Measure decision time
                start_time = time.time()
                
                # Run simulation
                result = self.simulator.run_simulation(tasks, strategy, sensitivity_test=True)
                
                end_time = time.time()
                decision_time = end_time - start_time
                
                # Collect results
                test_energy.append(result.total_energy)
                test_latency.append(result.total_latency)
                test_selections.append(result.device_selections)
                test_times.append(decision_time)
            
            # Aggregate results from this test
            total_energy.extend(test_energy)
            total_latency.extend(test_latency)
            device_selections.extend(test_selections)
            decision_times.extend(test_times)
        
        return {
            "energy": np.array(total_energy),
            "latency": np.array(total_latency),
            "selections": device_selections,
            "decision_times": np.array(decision_times)
        }
    
    def _calculate_improvements(self, strategy_results, baseline_results):
        """Calculate performance improvements over baseline"""
        
        # Energy improvement (lower is better)
        if np.mean(baseline_results["energy"]) != 0:
            energy_improvement = ((np.mean(baseline_results["energy"]) - 
                                np.mean(strategy_results["energy"])) / 
                               np.mean(baseline_results["energy"])) * 100
        else:
            energy_improvement = 0
        
        # Latency improvement (lower is better)
        if np.mean(baseline_results["latency"]) != 0:
            latency_improvement = ((np.mean(baseline_results["latency"]) - 
                                np.mean(strategy_results["latency"])) / 
                               np.mean(baseline_results["latency"])) * 100
        else:
            latency_improvement = 0
        
        # Decision time comparison (timing overhead)
        decision_time_ratio = np.mean(strategy_results["decision_times"]) / \
                            np.mean(baseline_results["decision_times"]) \
                            if np.mean(baseline_results["decision_times"]) > 0 else 1.0
        
        return {
            "energy_improvement": energy_improvement,
            "latency_improvement": latency_improvement,
            "decision_time_ratio": decision_time_ratio
        }
    
    def analyze_results(self, results):
        """Analyze the sensitivity results and generate visualizations"""
        
        # Create summary dataframe
        summary_data = []
        
        # Process baseline strategies first
        if "Baseline" in results:
            for strategy_name, data in results["Baseline"].items():
                summary_data.append({
                    "Strategy": strategy_name,
                    "Omega": "N/A",
                    "Energy (J)": np.mean(data["metrics"]["energy"]),
                    "Energy Std": np.std(data["metrics"]["energy"]),
                    "Latency (s)": np.mean(data["metrics"]["latency"]),
                    "Latency Std": np.std(data["metrics"]["latency"]),
                    "Decision Time (s)": np.mean(data["metrics"]["decision_times"]),
                    "Energy Improvement (%)": 0.0,
                    "Latency Improvement (%)": 0.0,
                    "Average Improvement (%)": 0.0
                })
        
        # Process FOFF strategy with different omega values
        if "FOFF" in results:
            for strategy_key, data in results["FOFF"].items():
                omega_str = "N/A"
                if "omega" in data:
                    omega_str = f"{data['omega'][0]},{data['omega'][1]}"
                
                summary_data.append({
                    "Strategy": data.get("strategy", "FOFF"),
                    "Omega": omega_str,
                    "Energy (J)": np.mean(data["metrics"]["energy"]),
                    "Energy Std": np.std(data["metrics"]["energy"]),
                    "Latency (s)": np.mean(data["metrics"]["latency"]),
                    "Latency Std": np.std(data["metrics"]["latency"]),
                    "Decision Time (s)": np.mean(data["metrics"]["decision_times"]),
                    "Energy Improvement (%)": data["improvements"]["energy_improvement"] if "improvements" in data else 0.0,
                    "Latency Improvement (%)": data["improvements"]["latency_improvement"] if "improvements" in data else 0.0,
                    "Average Improvement (%)": ((data["improvements"]["energy_improvement"] + 
                                               data["improvements"]["latency_improvement"]) / 2) if "improvements" in data else 0.0
                })
        
        df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        df.to_csv("sensitivity_results/sensitivity_summary.csv", index=False)
        
        # Create visualizations
        self._plot_heatmap_analysis(results)
        self._plot_best_configurations(df)
        self._analyze_device_selection(results)
        
        # Plot energy-latency tradeoff analysis
        self._plot_energy_latency_tradeoff(df)
        
        return df
    
    def _plot_heatmap_analysis(self, results):
        """Generate heatmaps to visualize parameter sensitivity"""
        
        try:
            # Wrap each plotting operation in try-except to ensure one failure doesn't stop all plots
            try:
                # Prepare data for energy efficiency heatmap
                energy_data_foff = self._prepare_heatmap_data(results["FOFF"], "energy_improvement")
                
                # Plot energy efficiency heatmap for FOFF
                plt.figure(figsize=(10, 8))
                sns.heatmap(energy_data_foff, annot=True, cmap="RdYlGn", fmt=".1f")
                plt.title("FOFF Energy Efficiency Improvement (%)")
                plt.xlabel("Weight 2 (Energy)")
                plt.ylabel("Weight 1 (Latency)")
                plt.tight_layout()
                plt.savefig("sensitivity_results/energy_heatmap_FOFF.png")
                plt.close()
            except Exception as e:
                self.logger.error(f"Error plotting FOFF Energy heatmap: {e}")
            
            try:
                # Prepare data for latency improvement heatmap
                latency_data_foff = self._prepare_heatmap_data(results["FOFF"], "latency_improvement")
                
                # Plot latency improvement heatmap for FOFF
                plt.figure(figsize=(10, 8))
                sns.heatmap(latency_data_foff, annot=True, cmap="RdYlGn", fmt=".1f")
                plt.title("FOFF Latency Improvement (%)")
                plt.xlabel("Weight 2 (Energy)")
                plt.ylabel("Weight 1 (Latency)")
                plt.tight_layout()
                # plt.savefig("sensitivity_results/latency_heatmap_FOFF.png")
                plt.savefig(f"sensitivity_results/latency_heatmap_FOFF.pdf", format="pdf")
                plt.close()
            except Exception as e:
                self.logger.error(f"Error plotting FOFF Latency heatmap: {e}")
                
        except Exception as e:
            self.logger.error(f"Error in heatmap analysis: {e}")
            self.logger.info("Continuing with other analyses...")
    
    def _prepare_heatmap_data(self, strategy_results, metric):
        """Prepare data for heatmap visualization"""
        
        # Create dataframe with importances as indices and columns
        heatmap_data = pd.DataFrame(index=self.importances, columns=self.importances)
        
        # Fill with zeros initially to avoid NaN values
        for idx in self.importances:
            for col in self.importances:
                heatmap_data.loc[idx, col] = 0.0
        
        for strategy_key, data in strategy_results.items():
            # Extract weights from omega
            if "omega" in data:
                w1, w2 = data["omega"]
                
                # Fill heatmap data
                if "improvements" in data and metric in data["improvements"]:
                    # Ensure value is a float
                    try:
                        value = float(data["improvements"][metric])
                        heatmap_data.loc[w1, w2] = value
                    except (ValueError, TypeError):
                        # If conversion fails, use 0.0
                        heatmap_data.loc[w1, w2] = 0.0
        
        # Explicitly convert all data to float
        heatmap_data = heatmap_data.astype(float)
        
        return heatmap_data
    
    def _plot_best_configurations(self, df):
        """Plot the best parameter configurations for different omega values"""
        
        # Filter FOFF strategies
        foff_df = df[df["Strategy"] == "FOFF"]
        
        # Sort by average improvement to find best configurations
        best_configs = foff_df.sort_values("Average Improvement (%)", ascending=False).head(5)
        
        # Plot comparison of best omega configurations
        plt.figure(figsize=(14, 10))
        
        # Create grouped bar chart
        x = np.arange(len(best_configs))
        width = 0.35
        
        plt.bar(x - width/2, best_configs["Energy Improvement (%)"], width, 
               label="Energy Improvement (%)")
        plt.bar(x + width/2, best_configs["Latency Improvement (%)"], width, 
               label="Latency Improvement (%)")
        
        plt.xlabel("FOFF Weight Configurations (ω)")
        plt.ylabel("Improvement Percentage (%)")
        plt.title("Best FOFF Parameter Configurations")
        
        # Create x-tick labels with omega values
        labels = [f"ω={row['Omega']}" for _, row in best_configs.iterrows()]
        plt.xticks(x, labels, rotation=45, ha="right")
        
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Add value labels
        for i, value in enumerate(best_configs["Energy Improvement (%)"]):
            plt.text(i - width/2, value + 0.5, f"{value:.1f}%", ha='center')
        
        for i, value in enumerate(best_configs["Latency Improvement (%)"]):
            plt.text(i + width/2, value + 0.5, f"{value:.1f}%", ha='center')
        
        plt.savefig(f"sensitivity_results/best_configurations.pdf", format="pdf")
        plt.close()
        
        # Save best configurations to CSV
        best_configs.to_csv("sensitivity_results/best_configurations.csv", index=False)
        
        # Also plot worst configurations for comparison
        worst_configs = foff_df.sort_values("Average Improvement (%)", ascending=True).head(5)
        
        plt.figure(figsize=(14, 10))
        x = np.arange(len(worst_configs))
        
        plt.bar(x - width/2, worst_configs["Energy Improvement (%)"], width, 
               label="Energy Improvement (%)")
        plt.bar(x + width/2, worst_configs["Latency Improvement (%)"], width, 
               label="Latency Improvement (%)")
        
        plt.xlabel("FOFF Weight Configurations (ω)")
        plt.ylabel("Improvement Percentage (%)")
        plt.title("Worst FOFF Parameter Configurations")
        
        labels = [f"ω={row['Omega']}" for _, row in worst_configs.iterrows()]
        plt.xticks(x, labels, rotation=45, ha="right")
        
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        for i, value in enumerate(worst_configs["Energy Improvement (%)"]):
            plt.text(i - width/2, value + 0.5, f"{value:.1f}%", ha='center')
        
        for i, value in enumerate(worst_configs["Latency Improvement (%)"]):
            plt.text(i + width/2, value + 0.5, f"{value:.1f}%", ha='center')
        
        plt.savefig(f"sensitivity_results/worst_configurations.pdf", format="pdf")

        plt.close()
        
    def _analyze_device_selection(self, results):
        """Analyze and visualize device selection patterns"""
        
        # Extract device selection data
        selection_data = defaultdict(list)
        
        # Handle baseline strategies
        for strategy_name, data in results["Baseline"].items():
            key = f"{strategy_name}"
            
            # Count device selections
            device_counts = defaultdict(int)
            
            for selections in data["metrics"]["selections"]:
                for device_id in selections:
                    device_counts[device_id] += 1
            
            selection_data[key].append(device_counts)
                
        # Handle FOFF strategy with different omega values
        for strategy_key, data in results["FOFF"].items():
            if "omega" in data:
                key = f"FOFF-{data['omega'][0]}_{data['omega'][1]}"
                
                # Count device selections
                device_counts = defaultdict(int)
                
                for selections in data["metrics"]["selections"]:
                    for device_id in selections:
                        device_counts[device_id] += 1
                
                selection_data[key].append(device_counts)
        
        # Create a composite selection pattern visualization
        plt.figure(figsize=(15, 10))
        
        # Identify key configurations to visualize
        key_configs = [
            # Baseline strategies
            "Random", "GTT", "GCF",
            # FOFF with different weight combinations
            "FOFF-vh_vl",  # Energy-focused
            "FOFF-vl_vh",  # Latency-focused
            "FOFF-m_m",    # Balanced
            "FOFF-h_l",    
            "FOFF-l_h",
            "FOFF-ml_mh"
        ]
        
        # Filter to only include configs that exist in the results
        available_configs = [config for config in key_configs if config in selection_data]
        
        # Prepare plot data
        num_configs = len(available_configs)
        num_devices = 6  # Assuming 6 devices (0-5)
        bar_width = 0.8 / num_configs
        
        # Create grouped bar chart of device selection percentages
        for i, config in enumerate(available_configs):
            # Aggregate device counts
            aggregated_counts = defaultdict(int)
            for counts in selection_data[config]:
                for device_id, count in counts.items():
                    aggregated_counts[device_id] += count
            
            # Convert to percentages
            total = sum(aggregated_counts.values())
            if total > 0:
                percentages = [
                    (aggregated_counts.get(device_id, 0) / total) * 100
                    for device_id in range(num_devices)
                ]
                
                # Calculate x positions for bars
                x = np.arange(num_devices)
                offset = (i - num_configs / 2) * bar_width + bar_width / 2
                
                # Plot bars
                plt.bar(x + offset, percentages, bar_width, label=config)
        
        plt.xlabel("Device ID")
        plt.ylabel("Selection Percentage (%)")
        plt.title("Device Selection Patterns for Different Algorithms and FOFF Weight Configurations")
        plt.xticks(np.arange(num_devices), [str(i) for i in range(num_devices)])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # plt.savefig("sensitivity_results/device_selection_patterns.png")
        plt.savefig(f"sensitivity_results/device_selection_patterns.pdf", format="pdf")

        plt.close()
    
    def _plot_energy_latency_tradeoff(self, df):
        """Plot energy-latency tradeoff for different FOFF configurations"""
        
        # Filter FOFF strategies
        foff_df = df[df["Strategy"] == "FOFF"]
        
        # Plot energy vs latency tradeoff
        plt.figure(figsize=(12, 10))
        
        # Extract omega first component for coloring
        omega_first = foff_df["Omega"].apply(lambda x: x.split(",")[0] if "," in x else "unknown")
        
        # Map linguistic terms to numeric values for marker size
        term_values = {
            "vl": 50,
            "l": 100,
            "ml": 150,
            "m": 200,
            "mh": 250,
            "h": 300,
            "vh": 350
        }
        
        # Extract omega second component for marker size
        omega_second = foff_df["Omega"].apply(lambda x: x.split(",")[1] if "," in x else "unknown")
        marker_sizes = omega_second.map(lambda x: term_values.get(x, 150))
        
        # Create a scatter plot
        scatter = plt.scatter(
            foff_df["Energy Improvement (%)"],
            foff_df["Latency Improvement (%)"],
            c=pd.factorize(omega_first)[0],
            s=marker_sizes,
            alpha=0.7,
            cmap="viridis"
        )
        
        # Add labels to each point
        for i, row in foff_df.iterrows():
            plt.text(
                row["Energy Improvement (%)"] + 0.2,
                row["Latency Improvement (%)"] + 0.2,
                f"ω={row['Omega']}",
                fontsize=8
            )
        
        # Add a legend for the first component colors
        legend1 = plt.legend(
            handles=scatter.legend_elements()[0],
            labels=sorted(set(omega_first)),
            title="Latency Weight",
            loc="upper left"
        )
        
        # Add a legend for the second component marker sizes
        sizes = [50, 150, 250, 350]
        labels = ["Very Low", "Medium Low", "Medium High", "Very High"]
        legend2 = plt.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', 
                     markerfacecolor='gray', markersize=np.sqrt(size/5)) for size in sizes],
            labels=labels,
            title="Energy Weight",
            loc="lower right"
        )
        
        plt.gca().add_artist(legend1)
        
        plt.xlabel("Energy Improvement (%)")
        plt.ylabel("Latency Improvement (%)")
        plt.title("Energy-Latency Trade-off for Different FOFF Weight Configurations")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"sensitivity_results/energy_latency_tradeoff.pdf", format="pdf")
        plt.close()

# Main execution
if __name__ == "__main__":
    # Configure logging
    logger = configure_logging(level=logging.INFO)
    logger.info("Starting FOFF Parameter Sensitivity Analysis")
    
    # Create configuration
    config = SimulationConfig()
    
    # Define device scenarios
    CHALLENGING_SCENARIO = [
        (2, 90, 0.0, np.array([451.36, 741.45]), 0),
        (4, 670, 0.0, np.array([642.61, 697.24]), 1),
        (2, 5, 0.0, np.array([722.94, 603.27]), 2),
        (4, 345, 0.0, np.array([670.94, 318.01]), 3),    
        (2, 88, 0.0, np.array([256.37, 503.55]), 4),    
        (0.005, 1310, 100.0, np.array([500.0, 500.0]), SimulationConfig.OBUI_INDEX),
    ]
    
    BALANCED_SCENARIO = [
        (2, 50, 0.0, np.array([451.36, 741.45]), 0),
        (3, 110, 0.0, np.array([256.37, 503.55]), 1),
        (2, 100, 0.0, np.array([722.94, 603.27]), 2),
        (4, 230, 0.0, np.array([670.94, 318.01]), 3),
        (2, 120, 0.0, np.array([642.61, 697.24]), 4),
        (0.005, 200, 100.0, np.array([500.0, 500.0]), SimulationConfig.OBUI_INDEX),    
    ]
    
    # Select scenario to use
    current_scenario = CHALLENGING_SCENARIO
    
    # Load task data
    tasks_shuffled = load("teste.csv")
    for i in range(len(tasks_shuffled)):
        for j in range(len(tasks_shuffled[0])):
            tasks_shuffled[i][j] = parse_task_string(tasks_shuffled[i][j])
    
    # Create sensitivity analyzer
    analyzer = FOFFSensitivityAnalyzer(config)
    
    # Run analysis
    logger.info("Running sensitivity analysis...")
    start_time = time.time()
    results = analyzer.run_analysis(current_scenario, tasks_shuffled[:10], num_tests=3)
    end_time = time.time()
    logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
    
    # Analyze and visualize results
    logger.info("Analyzing results and generating visualizations...")
    summary = analyzer.analyze_results(results)
    
    # Print summary of best configurations
    best_configs = summary[summary["Strategy"] == "FOFF"].sort_values(
        "Average Improvement (%)", ascending=False).head(5)
    
    logger.info("\nBest FOFF Parameter Configurations:")
    for idx, row in best_configs.iterrows():
        logger.info(f"  Omega: {row['Omega']}")
        logger.info(f"    Energy Improvement: {row['Energy Improvement (%)']:.2f}%")
        logger.info(f"    Latency Improvement: {row['Latency Improvement (%)']:.2f}%")
        logger.info(f"    Average Improvement: {row['Average Improvement (%)']:.2f}%")
    
    logger.info("\nSensitivity analysis complete. Results saved to 'sensitivity_results' directory.")