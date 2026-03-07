import csv
import json
import os

def save(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

def load(filename):
    tasks_from_csv = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # 'row' será uma lista de strings que representa cada linha do CSV
            tasks_from_csv.append(row)
    return tasks_from_csv

def save_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_data(filename):
    if os.path.exists(filename) == False:
        return {
            'energy_consumption': dict(),
            'processed_time': dict(),
            'processed_cycles': dict(),    
        }
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
