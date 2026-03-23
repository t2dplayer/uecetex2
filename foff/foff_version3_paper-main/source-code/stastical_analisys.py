import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import argparse
import sys


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import argparse

def load_data(file_path):
    """Load the CSV data into a pandas DataFrame"""
    return pd.read_csv(file_path)

def perform_wilcoxon_tests(df, metric_type):
    """
    Perform Wilcoxon signed-rank tests between all pairs of algorithms
    for a specific metric type, with special handling for identical data
    """
    # Filter data for the specified metric
    metric_df = df[df['metric_type'] == metric_type]
    
    # Get unique algorithms
    algorithms = metric_df['algorithm'].unique()
    
    # Create a dictionary to store results
    results = {}
    
    # Perform pairwise comparisons
    for alg1, alg2 in itertools.combinations(algorithms, 2):
        # Extract values for each algorithm
        alg1_values = metric_df[metric_df['algorithm'] == alg1]['value'].values
        alg2_values = metric_df[metric_df['algorithm'] == alg2]['value'].values
        
        # Skip if sample sizes don't match
        if len(alg1_values) != len(alg2_values):
            print(f"Skipping {alg1} vs {alg2} due to unequal sample sizes")
            continue
        
        # Calculate differences
        differences = alg1_values - alg2_values
        
        # Check if all differences are zero
        if np.all(differences == 0):
            print(f"Note: {alg1} and {alg2} have identical values for all samples")
            results[f"{alg1} vs {alg2}"] = {
                'statistic': 0,
                'p_value': 1.0,  # p-value of 1 indicates no difference
                'significant': False,
                'mean_diff': 0,
                'identical': True
            }
            continue
        
        # Remove zero differences for Wilcoxon test
        non_zero_diff = differences[differences != 0]
        
        # If there are no non-zero differences, skip this pair
        if len(non_zero_diff) == 0:
            results[f"{alg1} vs {alg2}"] = {
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'mean_diff': 0,
                'identical': True
            }
            continue
            
        try:
            # Perform Wilcoxon signed-rank test
            stat, p_value = stats.wilcoxon(alg1_values, alg2_values)
            
            # Store results
            results[f"{alg1} vs {alg2}"] = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_diff': np.mean(alg1_values) - np.mean(alg2_values),
                'identical': False
            }
        except ValueError as e:
            print(f"Error performing Wilcoxon test for {alg1} vs {alg2}: {e}")
            # Try an alternative approach - Mann-Whitney U test (less powerful but works with zeros)
            try:
                stat, p_value = stats.mannwhitneyu(alg1_values, alg2_values)
                print(f"Used Mann-Whitney U test instead for {alg1} vs {alg2}")
                results[f"{alg1} vs {alg2}"] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'mean_diff': np.mean(alg1_values) - np.mean(alg2_values),
                    'test_used': 'Mann-Whitney U',
                    'identical': False
                }
            except Exception as e2:
                print(f"Alternative test also failed: {e2}")
                # Just record the means and indicate test failure
                results[f"{alg1} vs {alg2}"] = {
                    'statistic': None,
                    'p_value': None,
                    'significant': None,
                    'mean_diff': np.mean(alg1_values) - np.mean(alg2_values),
                    'test_failed': True,
                    'identical': False
                }
    
    return results

def create_summary_table(wilcoxon_results, metric_type):
    """Create a summary table of statistical test results"""
    # Check if we have any results
    if not wilcoxon_results:
        print(f"\nNo statistical test results for {metric_type} (possibly due to identical data)")
        return None
    
    # Create a dataframe from the results
    data = []
    for comparison, res in wilcoxon_results.items():
        row = {
            'Comparison': comparison,
            'Mean Difference': res['mean_diff']
        }
        
        # Add test statistics if available
        if 'test_failed' in res and res['test_failed']:
            row['Test'] = 'Failed'
            row['Statistic'] = 'N/A'
            row['p-value'] = 'N/A'
            row['Significant (p<0.05)'] = 'N/A'
        elif 'identical' in res and res['identical']:
            row['Test'] = 'N/A (Identical)'
            row['Statistic'] = 0
            row['p-value'] = 1.0
            row['Significant (p<0.05)'] = False
        else:
            row['Test'] = res.get('test_used', 'Wilcoxon')
            row['Statistic'] = res['statistic']
            row['p-value'] = res['p_value']
            row['Significant (p<0.05)'] = res['significant']
        
        data.append(row)
    
    results_df = pd.DataFrame(data)
    
    # Sort by p-value where available, then by mean difference
    if 'p-value' in results_df.columns and not all(results_df['p-value'].astype(str) == 'N/A'):
        # Convert 'N/A' to NaN for sorting
        p_values = pd.to_numeric(results_df['p-value'], errors='coerce')
        results_df = results_df.iloc[np.argsort(p_values.fillna(2))]
    else:
        # Sort by absolute mean difference
        results_df = results_df.iloc[np.argsort(np.abs(results_df['Mean Difference'])).values[::-1]]
    
    print(f"\nStatistical Test Results for {metric_type}:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string(index=False))
    
    return results_df

def visualize_algorithm_performance(df, metric_type):
    """Create box plots to compare algorithm performance"""
    # Filter data for the specified metric
    metric_df = df[df['metric_type'] == metric_type]
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='algorithm', y='value', data=metric_df)
    plt.title(f'Algorithm Performance Comparison - {metric_type}')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{metric_type}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{metric_type}_boxplot.png')
    
    # Also create a violin plot for more detailed distribution view
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='algorithm', y='value', data=metric_df)
    plt.title(f'Algorithm Performance Distribution - {metric_type}')
    plt.xlabel('Algorithm')
    plt.ylabel(f'{metric_type}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{metric_type}_violinplot.png')

def calculate_effect_size(df, metric_type):
    """Calculate effect size (Cohen's d) between algorithm pairs"""
    # Filter data for the specified metric
    metric_df = df[df['metric_type'] == metric_type]
    
    # Get unique algorithms
    algorithms = metric_df['algorithm'].unique()
    
    # Create a dictionary to store results
    effect_sizes = {}
    
    # Calculate effect size for each pair
    for alg1, alg2 in itertools.combinations(algorithms, 2):
        # Extract values for each algorithm
        alg1_values = metric_df[metric_df['algorithm'] == alg1]['value'].values
        alg2_values = metric_df[metric_df['algorithm'] == alg2]['value'].values
        
        # Skip if sample sizes don't match
        if len(alg1_values) != len(alg2_values):
            continue
        
        # Check if all values are identical
        if np.array_equal(alg1_values, alg2_values):
            effect_sizes[f"{alg1} vs {alg2}"] = 0
            continue
        
        # Calculate means and standard deviations
        mean1, mean2 = np.mean(alg1_values), np.mean(alg2_values)
        std1, std2 = np.std(alg1_values, ddof=1), np.std(alg2_values, ddof=1)
        
        # Check for standard deviation of zero to avoid division by zero
        if std1 == 0 and std2 == 0:
            if mean1 == mean2:
                effect_sizes[f"{alg1} vs {alg2}"] = 0
            else:
                # If means differ but no variance, effect is theoretically infinite
                # We'll use a very large number instead
                effect_sizes[f"{alg1} vs {alg2}"] = float('inf') if mean1 > mean2 else float('-inf')
            continue
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((len(alg1_values) - 1) * std1**2 + (len(alg2_values) - 1) * std2**2) / 
                            (len(alg1_values) + len(alg2_values) - 2))
        
        # Handle case where pooled_std is zero
        if pooled_std == 0:
            if mean1 == mean2:
                effect_sizes[f"{alg1} vs {alg2}"] = 0
            else:
                # If means differ but no variance, effect is theoretically infinite
                effect_sizes[f"{alg1} vs {alg2}"] = float('inf') if mean1 > mean2 else float('-inf')
            continue
        
        # Cohen's d
        cohen_d = (mean1 - mean2) / pooled_std
        
        effect_sizes[f"{alg1} vs {alg2}"] = cohen_d
    
    return effect_sizes

def find_best_algorithm(df, metric_type, lower_is_better=True):
    """
    Determine the best performing algorithm based on mean performance
    and statistical significance
    """
    # Filter data for the specified metric
    metric_df = df[df['metric_type'] == metric_type]
    
    # Calculate mean performance for each algorithm
    algorithm_means = metric_df.groupby('algorithm')['value'].mean().reset_index()
    
    # Sort based on whether lower or higher values are better
    if lower_is_better:
        algorithm_means = algorithm_means.sort_values('value')
    else:
        algorithm_means = algorithm_means.sort_values('value', ascending=False)
    
    print(f"\nAlgorithm ranking for {metric_type} ({'lower' if lower_is_better else 'higher'} is better):")
    for i, (_, row) in enumerate(algorithm_means.iterrows()):
        print(f"{i+1}. {row['algorithm']}: {row['value']:.6f}")
    
    return algorithm_means

def check_identical_data(df, metric_type):
    """Check if there are algorithms with identical data for a given metric"""
    # Filter data for the specified metric
    metric_df = df[df['metric_type'] == metric_type]
    
    # Get unique algorithms
    algorithms = metric_df['algorithm'].unique()
    
    identical_pairs = []
    
    # Check each pair of algorithms
    for alg1, alg2 in itertools.combinations(algorithms, 2):
        # Extract values for each algorithm
        alg1_values = metric_df[metric_df['algorithm'] == alg1]['value'].reset_index(drop=True)
        alg2_values = metric_df[metric_df['algorithm'] == alg2]['value'].reset_index(drop=True)
        
        # Skip if sample sizes don't match
        if len(alg1_values) != len(alg2_values):
            continue
        
        # Check if values are identical
        if alg1_values.equals(alg2_values):
            identical_pairs.append((alg1, alg2))
    
    if identical_pairs:
        print(f"\nIdentical algorithm data found for {metric_type}:")
        for alg1, alg2 in identical_pairs:
            print(f"  - {alg1} and {alg2} have identical values")
    
    return identical_pairs

def main(file_path):
    # Load the data
    df = load_data(file_path)
    
    # Print basic information about the dataset
    print("Dataset Information:")
    print(f"Total records: {len(df)}")
    print(f"Unique initial queue waiting time: {df['initial_queue_waiting_time'].nunique()}")
    print(f"Algorithms: {', '.join(df['algorithm'].unique())}")
    print(f"Metric types: {', '.join(df['metric_type'].unique())}")
    
    # Analyze each metric type
    for metric_type in df['metric_type'].unique():
        print(f"\n{'='*50}")
        print(f"Analysis for {metric_type}")
        print(f"{'='*50}")
        
        # Check for identical data
        identical_pairs = check_identical_data(df, metric_type)
        
        # Perform statistical tests
        wilcoxon_results = perform_wilcoxon_tests(df, metric_type)
        
        # Create summary table
        summary_df = create_summary_table(wilcoxon_results, metric_type)
        
        # Calculate effect sizes
        effect_sizes = calculate_effect_size(df, metric_type)
        
        # Print effect sizes
        print("\nEffect Sizes (Cohen's d):")
        for comparison, effect_size in effect_sizes.items():
            if effect_size == float('inf'):
                print(f"{comparison}: Infinite (means differ but no variance)")
                continue
            elif effect_size == float('-inf'):
                print(f"{comparison}: -Infinite (means differ but no variance)")
                continue
                
            magnitude = ""
            if abs(effect_size) < 0.2:
                magnitude = "negligible"
            elif abs(effect_size) < 0.5:
                magnitude = "small"
            elif abs(effect_size) < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            print(f"{comparison}: {effect_size:.4f} ({magnitude})")
        
        # Determine best algorithm
        # For energy_consumption and processed_time, lower is better
        # For other metrics, you may need to adjust this
        lower_is_better = True
        if metric_type in ["energy_consumption", "processed_time", "processed_cycles"]:
            lower_is_better = True
        else:
            lower_is_better = False
            
        best_algorithms = find_best_algorithm(df, metric_type, lower_is_better)
        
        # Create visualizations
        visualize_algorithm_performance(df, metric_type)
        
        print(f"\nVisualizations for {metric_type} saved as '{metric_type}_boxplot.png' and '{metric_type}_violinplot.png'")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze algorithm performance with statistical tests')
    parser.add_argument('filename', help='Path to the CSV file containing algorithm performance data')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with provided filename
    main(args.filename)