from device import QuantumDevice
from simulator import QuantumSimulation
from policy import (
    random_device,
    random_load_weighted,
    least_busy,
    fidelity_weighted,
    qoncord,
    best_fidelity,
    eqc_scheduling,
)

import numpy as np

from tqdm import tqdm
import pandas as pd

from plot_queue_sim_result import plot_queue_sim


np.random.seed(20231126)

policies = {
    "random": random_device,
    "random_load_weighted": random_load_weighted,
    "least_busy": least_busy,
    "fidelity_weighted": fidelity_weighted,
    "qoncord": qoncord,
    "best_fidelity": best_fidelity,
    "eqc_scheduling": eqc_scheduling,
}


def get_devices(n_devices, fidelity_range=(0.3, 0.9)):
    fidelities = np.linspace(*fidelity_range, n_devices)[::-1]

    devices = [QuantumDevice(i, fidelities[i]) for i in range(n_devices)]
    
    return devices

def get_jobs_attributes(n_jobs, runtime_ratio):
    is_runtime = []
    circuit_time = []
    for _ in range(n_jobs):
        if np.random.random() < runtime_ratio:
            is_runtime.append(True)
            circuit_time.append(0)
        else:
            is_runtime.append(False)
            circuit_time.append(np.random.randint(3, 10))

    return is_runtime, circuit_time


n_devices = 10

n_jobs = 1000

runtime_ratios = np.linspace(0.1, 0.9, 9)
results = {
    "runtime_ratio": [],
    "policy": [],
    "max_device_completion_time": [],
    "device_vacancy_times": [],
    "average_fidelity": [],
    "average_shared_fidelity": [],
    "average_runtime_fidelity": [],
    "total_job_time": [],
    "total_circuits": [],
}

for runtime_ratio in tqdm(runtime_ratios):
    
    is_runtime, circuit_time = get_jobs_attributes(n_jobs, runtime_ratio)
    
    policy_fidelities = []
    policy_shared_fidelities = []
    policy_runtime_fidelities = []

    for policy in tqdm(policies, leave=False):
        devices = get_devices(n_devices)

        if policy == "eqc_scheduling":
            device_loads, total_job_time = policies[policy](
                is_runtime, circuit_time, devices
            )
            total_circuits = sum([100 * 2 if is_runtime_sub else 1 for is_runtime_sub in is_runtime])

            max_device_completion_time = max(device_loads)
            device_vacancy_times = [
                float(max_device_completion_time - t) for t in device_loads
            ]
            average_fidelity = np.median(policy_fidelities)
            shared_fidelity = np.median(policy_shared_fidelities)
            runtime_fidelity = np.median(policy_runtime_fidelities)

        else:
            jobs = policies[policy](is_runtime, circuit_time, devices)
            total_circuits = sum([100 if is_runtime_sub else 1 for is_runtime_sub in is_runtime])

            simulation = QuantumSimulation(devices, jobs)
            simulation.run_simulation()

            device_completion_times = np.zeros(n_devices)

            total_job_time = 0

            for job in jobs:
                device_completion_times[job.device_id] = max(
                    device_completion_times[job.device_id], job.end_times[-1]
                )
                total_job_time += sum(
                    [end - start for start, end in zip(job.start_times, job.end_times)]
                )

            max_device_completion_time = max(device_completion_times)

            device_vacancy_times = [
                float(max_device_completion_time - t) for t in device_completion_times
            ]
            average_shared_fidelity = np.mean(
                [job.fidelity for job in jobs if not job.runtime]
            )
            average_runtime_fidelity = np.mean(
                [job.fidelity for job in jobs if job.runtime]
            )
            average_fidelity = np.mean([job.fidelity for job in jobs])
            
            
            policy_fidelities.append(average_fidelity)
            policy_shared_fidelities.append(average_shared_fidelity)
            policy_runtime_fidelities.append(average_runtime_fidelity)

        results["runtime_ratio"].append(runtime_ratio)
        results["policy"].append(policy)
        results["max_device_completion_time"].append(max_device_completion_time)
        results["device_vacancy_times"].append(device_vacancy_times)
        results["average_fidelity"].append(average_fidelity)
        results["average_shared_fidelity"].append(average_shared_fidelity)
        results["average_runtime_fidelity"].append(average_runtime_fidelity)
        results["total_job_time"].append(total_job_time)
        results["total_circuits"].append(total_circuits)

# Convert results to a DataFrame and plot the results
df = pd.DataFrame(results)
df.to_csv("queue_sim_results.csv", index=False)

# plot_queue_sim(df)