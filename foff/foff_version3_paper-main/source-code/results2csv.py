import json
import csv
import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path

def flatten_json_to_csv(json_file_path, csv_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Initialize lists to store data
    records = []
    
    # Extract data for energy consumption
    for initial_queue_waiting_time, device_data in data["energy_consumption"].items():
        for algorithm, algorithm_data in device_data.items():
            values = algorithm_data["value"]
            
            for i, value in enumerate(values):
                record = {
                    "initial_queue_waiting_time": initial_queue_waiting_time,
                    "algorithm": algorithm,
                    "metric_type": "energy_consumption",
                    "sample_id": i,
                    "value": value
                }
                records.append(record)
    
    # Extract data for processed_time
    for initial_queue_waiting_time, device_data in data["processed_time"].items():
        for algorithm, algorithm_data in device_data.items():
            values = algorithm_data["value"]
            
            for i, value in enumerate(values):
                record = {
                    "initial_queue_waiting_time": initial_queue_waiting_time,
                    "algorithm": algorithm,
                    "metric_type": "processed_time",
                    "sample_id": i,
                    "value": value
                }
                records.append(record)
    
    # Extract data for processed_cycles
    for initial_queue_waiting_time, device_data in data["processed_cycles"].items():
        for algorithm, algorithm_data in device_data.items():
            values = algorithm_data["value"]
            
            for i, value in enumerate(values):
                record = {
                    "initial_queue_waiting_time": initial_queue_waiting_time,
                    "algorithm": algorithm,
                    "metric_type": "processed_cycles",
                    "sample_id": i,
                    "value": value
                }
                records.append(record)
    
    # Convert to pandas DataFrame for easy CSV export
    df = pd.DataFrame(records)
    
    # Write to CSV
    df.to_csv(csv_file_path, index=False)
    
    print(f"Successfully converted JSON to CSV. Output saved to {csv_file_path}")
    print(f"Total records: {len(records)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Results converter')
    parser.add_argument(
        '--filename',
        type=str,
    )
    args = parser.parse_args()
    if len(sys.argv) != 3:
        print(sys.argv[2], len(sys.argv))
        raise "Erro"
    # Usage    
    csv_file_name = Path(args.filename).stem + ".csv"
    flatten_json_to_csv(args.filename, csv_file_name)