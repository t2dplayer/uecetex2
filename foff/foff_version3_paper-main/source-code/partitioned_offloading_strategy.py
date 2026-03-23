"""
Partitioned Offloading Strategy for Vehicular Edge Computing
Based on the paper: "Online Partitioned Scheduling over RSU for Computation Offloading in Vehicular Edge Computing"

This implementation provides two strategies:
1. FP-Sched (Fully Partitioned Scheduling) - tasks must be executed entirely on a single device
2. HP-Sched (Hybrid Partitioned Scheduling) - tasks can be split into subtasks and executed across multiple devices
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Set
from offloading_strategy import OffloadingStrategy
from device_manager import Device, DeviceManager
from task_manager import Task
import logging

class PartitionedTask:
    """Represents a task that can be partitioned into subtasks."""
    def __init__(self, task: Task, num_subtasks: int = None):
        self.task = task
        self.subtasks = []
        self.create_subtasks(num_subtasks)
        self.scheduled = False
        self.scheduled_subtasks = 0

    def create_subtasks(self, num_subtasks: int = None):
        """Splits the task into subtasks with dependencies."""
        if num_subtasks is None:
            # Randomly decide number of subtasks (between 1 and 5)
            num_subtasks = random.randint(1, min(5, int(self.task.data_size)))
        
        # Ensure at least 1 subtask
        num_subtasks = max(1, num_subtasks)
        
        # Split the task's CPU cycles and data size among subtasks
        total_size = self.task.data_size
        total_cycles = self.task.cpu_cycles
        
        # Create subtasks with proportional sizes and dependencies
        sizes = []
        for i in range(num_subtasks - 1):
            # Randomly assign a portion of the remaining size
            size = random.uniform(0.1, 0.5) * total_size / (num_subtasks - i)
            sizes.append(size)
            total_size -= size
        
        # Last subtask gets the remainder
        sizes.append(total_size)
        
        # Create subtasks with dependency chain (linear for simplicity)
        self.subtasks = []
        for i in range(num_subtasks):
            subtask = {
                'id': i,
                'data_size': sizes[i],
                'cpu_cycles': total_cycles,  # Each subtask has same cycles per MB
                'dependencies': [i-1] if i > 0 else [],  # Simple linear dependency
                'scheduled': False,
                'device_id': None
            }
            self.subtasks.append(subtask)

    def is_subtask_ready(self, subtask_id: int) -> bool:
        """Check if a subtask is ready to be scheduled (all dependencies are satisfied)."""
        subtask = self.subtasks[subtask_id]
        for dep_id in subtask['dependencies']:
            if not self.subtasks[dep_id]['scheduled']:
                return False
        return True

    def get_next_schedulable_subtask(self) -> int:
        """Get the next subtask that can be scheduled."""
        for i, subtask in enumerate(self.subtasks):
            if not subtask['scheduled'] and self.is_subtask_ready(i):
                return i
        return -1  # No schedulable subtask found

    def is_fully_scheduled(self) -> bool:
        """Check if all subtasks are scheduled."""
        return all(subtask['scheduled'] for subtask in self.subtasks)


class FPSched(OffloadingStrategy):
    """
    Fully Partitioned Scheduling Strategy (FP-Sched)
    
    Tasks must be executed entirely on a single device (RSU).
    This strategy uses First Fit Decreasing Utilization (FFDU) to schedule tasks.
    """
    def __init__(self, simulator, time_slot_length: float = 10.0):
        super().__init__(simulator)
        self.time_slot_length = time_slot_length
        self.current_time_slot = 0
        self.scheduled_tasks = set()  # Track which tasks are scheduled
        self.device_allocations = {}  # Map device_id to set of scheduled task_ids
        self.logger = logging.getLogger(__name__)

    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager) -> tuple:
        """Execute the FP-Sched algorithm for the current task."""
        task = tasks[current_task_index]
        
        # If task is already scheduled, no action needed
        if current_task_index in self.scheduled_tasks:
            return None, None
        
        # Get current time slot boundaries
        time_slot_start = self.current_time_slot * self.time_slot_length
        time_slot_end = (self.current_time_slot + 1) * self.time_slot_length
        
        # Sort devices by increasing order of entry time (t_in)
        devices_with_times = []
        for device in device_manager.devices:
            # Calculate approximate t_in and t_out based on device position and task arrival time
            obui = device_manager.get_device(self.simulator.obui_index)
            distance = np.linalg.norm(device.position - obui.position)
            
            if distance <= 0:
                continue  # Skip self-offloading (device to itself)
                
            # Note: This is an approximation; the actual calculation would depend on vehicle routes
            t_in = max(task.arrival_time, self.current_time_slot * self.time_slot_length)
            t_out = t_in + (self.simulator.wireless.calculate_transmission_time(distance, task.data_size) * 2) + device.queue_time + (task.data_size * task.cpu_cycles) / (device.frequency_ghz * 1e9)
            
            # Only consider devices that are available within the current time slot
            if t_in <= time_slot_end and t_out >= time_slot_start:
                devices_with_times.append((device, t_in, t_out))
        
        # Sort by entry time
        devices_with_times.sort(key=lambda x: x[1])
        
        # Try to find a suitable device for the task
        for device, t_in, t_out in devices_with_times:
            # Calculate available processing time in the device
            available_time = t_out - t_in
            
            # Calculate data transfer time
            obui = device_manager.get_device(self.simulator.obui_index)
            distance = np.linalg.norm(device.position - obui.position)
            data_transfer_time = self.simulator.wireless.calculate_transmission_time(distance, task.data_size)
            
            # Calculate task execution time
            execution_time = (task.data_size * task.cpu_cycles) / (device.frequency_ghz * 1e9)
            
            # Calculate total time required
            total_required_time = data_transfer_time * 2 + execution_time  # Upload + execution + download
            
            # Check if the device has enough time to process the task
            if available_time >= total_required_time:
                # Check if there's enough remaining capacity in this time slot
                if device.device_id not in self.device_allocations:
                    self.device_allocations[device.device_id] = set()
                
                # Simple utilization check (assuming linear capacity)
                # In a more detailed implementation, this would check against maximum utilization
                device_utilization = sum(tasks[tid].data_size * tasks[tid].cpu_cycles for tid in self.device_allocations[device.device_id]) / (device.frequency_ghz * 1e9)
                
                if device_utilization + (task.data_size * task.cpu_cycles) / (device.frequency_ghz * 1e9) <= self.time_slot_length:
                    # Allocate task to this device
                    self.device_allocations[device.device_id].add(current_task_index)
                    self.scheduled_tasks.add(current_task_index)
                    self.logger.debug(f"FP-Sched: Task {current_task_index} scheduled to device {device.device_id}")
                    return device, None
        
        # No suitable device found
        return None, None
    
    def update_time_slot(self, new_time_slot: int):
        """Update the current time slot."""
        if new_time_slot > self.current_time_slot:
            # Clear allocations when moving to a new time slot
            self.current_time_slot = new_time_slot
            self.device_allocations = {}
    
    def __str__(self):
        return 'FP-Sched'


class HPSched(OffloadingStrategy):
    """
    Hybrid Partitioned Scheduling Strategy (HP-Sched)
    
    Tasks can be split into subtasks and executed across multiple devices (RSUs).
    This strategy combines FP-Sched with Semi-Partitioned scheduling.
    """
    def __init__(self, simulator, time_slot_length: float = 500.0, migration_penalty: float = 1.0):
        super().__init__(simulator)
        self.time_slot_length = time_slot_length
        self.current_time_slot = 0
        self.migration_penalty = migration_penalty  # Penalty for migrating between devices
        self.partitioned_tasks = {}  # Map task_id to PartitionedTask
        self.device_allocations = {}  # Map device_id to set of scheduled subtasks
        self.scheduled_tasks = set()  # Track which tasks are fully scheduled
        self.fp_strategy = FPSched(simulator, time_slot_length)
        self.logger = logging.getLogger(__name__)

    def execute(self, tasks, current_task_index: int, device_manager: DeviceManager) -> tuple:
        """Execute the HP-Sched algorithm for the current task."""
        # First, try the fully partitioned strategy
        device, _ = self.fp_strategy.execute(tasks, current_task_index, device_manager)
        
        if device is not None:
            self.scheduled_tasks.add(current_task_index)
            return device, None
        
        # If fully partitioned fails, try semi-partitioned strategy
        task = tasks[current_task_index]
        
        # Create partitioned task if not exists
        if current_task_index not in self.partitioned_tasks:
            self.partitioned_tasks[current_task_index] = PartitionedTask(task)
        
        partitioned_task = self.partitioned_tasks[current_task_index]
        
        # If task is already fully scheduled, no action needed
        if partitioned_task.is_fully_scheduled():
            self.scheduled_tasks.add(current_task_index)
            return None, None
        
        # Get current time slot boundaries
        time_slot_start = self.current_time_slot * self.time_slot_length
        time_slot_end = (self.current_time_slot + 1) * self.time_slot_length
        
        # Get the next subtask that can be scheduled
        subtask_id = partitioned_task.get_next_schedulable_subtask()
        if subtask_id == -1:
            return None, None  # No schedulable subtask found
        
        subtask = partitioned_task.subtasks[subtask_id]
        
        # Sort devices by increasing order of entry time (t_in)
        devices_with_times = []
        for device in device_manager.devices:
            # Calculate approximate t_in and t_out based on device position and task arrival time
            obui = device_manager.get_device(self.simulator.obui_index)
            distance = np.linalg.norm(device.position - obui.position)
            
            if distance <= 0:
                continue  # Skip self-offloading (device to itself)
                
            # Note: This is an approximation; the actual calculation would depend on vehicle routes
            t_in = max(task.arrival_time, self.current_time_slot * self.time_slot_length)
            
            # For subtasks, we use proportional execution time based on data size
            subtask_execution_time = (subtask['data_size'] * task.cpu_cycles) / (device.frequency_ghz * 1e9)
            subtask_transfer_time = self.simulator.wireless.calculate_transmission_time(distance, subtask['data_size'])
            
            # Add migration penalty for subtasks with dependencies
            migration_time = self.migration_penalty if subtask['dependencies'] else 0
            
            t_out = t_in + (subtask_transfer_time * 2) + subtask_execution_time + migration_time + device.queue_time
            
            # Only consider devices that are available within the current time slot
            if t_in <= time_slot_end and t_out >= time_slot_start:
                devices_with_times.append((device, t_in, t_out))
        
        # Sort by entry time
        devices_with_times.sort(key=lambda x: x[1])
        
        # Try to find a suitable device for the subtask
        for device, t_in, t_out in devices_with_times:
            # Calculate available processing time in the device
            available_time = t_out - t_in
            
            # Calculate data transfer time
            obui = device_manager.get_device(self.simulator.obui_index)
            distance = np.linalg.norm(device.position - obui.position)
            data_transfer_time = self.simulator.wireless.calculate_transmission_time(distance, subtask['data_size'])
            
            # Calculate subtask execution time
            execution_time = (subtask['data_size'] * task.cpu_cycles) / (device.frequency_ghz * 1e9)
            
            # Add migration penalty for subtasks with dependencies
            migration_time = self.migration_penalty if subtask['dependencies'] else 0
            
            # Calculate total time required
            total_required_time = data_transfer_time * 2 + execution_time + migration_time  # Upload + execution + download + migration
            
            # Check if the device has enough time to process the subtask
            if available_time >= total_required_time:
                # Check if there's enough remaining capacity in this time slot
                if device.device_id not in self.device_allocations:
                    self.device_allocations[device.device_id] = set()
                
                # Simple utilization check (assuming linear capacity)
                # Here we would need to track the actual utilization of each device more precisely
                
                # Schedule subtask to this device
                self.device_allocations[device.device_id].add((current_task_index, subtask_id))
                subtask['scheduled'] = True
                subtask['device_id'] = device.device_id
                partitioned_task.scheduled_subtasks += 1
                
                self.logger.debug(f"HP-Sched: Task {current_task_index} Subtask {subtask_id} scheduled to device {device.device_id}")
                
                # Check if all subtasks are scheduled
                if partitioned_task.is_fully_scheduled():
                    self.scheduled_tasks.add(current_task_index)
                
                # Return the device that will handle the "main" task or the current subtask
                # This is a simplification for the simulator
                return device, (subtask_id, partitioned_task.subtasks)
        
        # No suitable device found for this subtask
        return None, None
    
    def update_time_slot(self, new_time_slot: int):
        """Update the current time slot."""
        if new_time_slot > self.current_time_slot:
            # Clear allocations when moving to a new time slot
            self.current_time_slot = new_time_slot
            self.device_allocations = {}
            self.fp_strategy.update_time_slot(new_time_slot)
    
    def __str__(self):
        return 'HP-Sched'