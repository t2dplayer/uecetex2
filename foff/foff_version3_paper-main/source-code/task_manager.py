from dataclasses import dataclass
import numpy as np
import re

@dataclass
class Task:
    data_size: float # Megabytes
    cpu_cycles: float  # CPY cycles per Megabyte
    task_id: int
    arrival_time: float

def parse_task_string(task_str: str) -> Task:
    """
    Recebe uma string no formato:
        "Task(data_size=..., cpu_cycles=..., task_id=..., arrival_time=...)"
    e devolve um objeto Task.
    """
    # Opção 1: Usar expressão regular
    pattern = (
        r"Task\s*\(\s*data_size\s*=\s*(?P<data_size>\d+)\s*,\s*"
        r"cpu_cycles\s*=\s*(?P<cpu_cycles>\d+)\s*,\s*"
        r"task_id\s*=\s*(?P<task_id>\d+)\s*,\s*"
        r"arrival_time\s*=\s*(?P<arrival_time>[0-9\.]+)\s*\)"
    )
    match = re.match(pattern, task_str.strip())
    if not match:
        raise ValueError(f"String '{task_str}' não corresponde ao padrão esperado.")

    data_size = int(match.group("data_size"))
    cpu_cycles = int(match.group("cpu_cycles"))
    task_id = int(match.group("task_id"))
    arrival_time = float(match.group("arrival_time"))

    return Task(data_size, cpu_cycles, task_id, arrival_time)

class TaskManager:
    def __init__(self, config: list):
        self.tasks = [self._parse_task(d) for d in config]
    @staticmethod
    def _parse_task(raw_data: tuple) -> Task:
        return Task(
            data_size=raw_data[0],
            cpu_cycles=raw_data[1],
            task_id=raw_data[2],
            arrival_time=raw_data[3],
        )

    def get_task(self, task_id: int) -> Task:
        return next(d for d in self.tasks if d.task_id == task_id)