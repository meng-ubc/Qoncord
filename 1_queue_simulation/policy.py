from job import QuantumJob
import numpy as np

def generate_shared_job(job_id, device_id, execution_time):
    return QuantumJob(job_id=job_id, device_id=device_id, time_per_job=execution_time)


def generate_runtime_job(job_id, device_id, is_qoncord):
    return QuantumJob(
        job_id=job_id,
        device_id=device_id,
        runtime=True,
        qoncord=is_qoncord,
        runtime_cooldown=2,
    )


def generate_job(job_id, device_id, is_runtime, execution_time, is_qoncord):
    if is_runtime:
        return generate_runtime_job(job_id, device_id, is_qoncord)
    else:
        return generate_shared_job(job_id, device_id, execution_time)


def random_device(is_runtime_all, circuit_times, devices):
    jobs = []
    for i, (is_runtime, execution_time) in enumerate(
        zip(is_runtime_all, circuit_times)
    ):
        device_id = np.random.choice(len(devices))
        jobs.append(generate_job(i, device_id, is_runtime, execution_time, False))

    return jobs


def random_load_weighted(is_runtime_all, circuit_times, devices):
    jobs = []
    device_loads = [0] * len(devices)
    for i, (is_runtime, execution_time) in enumerate(
        zip(is_runtime_all, circuit_times)
    ):
        inverse_loads = 1 / (
            np.array(device_loads) + 1
        )  # Adding 1 to avoid division by zero
        normalized_inverse_loads = inverse_loads / inverse_loads.sum()

        device_id = np.random.choice(len(devices), p=normalized_inverse_loads)

        jobs.append(generate_job(i, device_id, is_runtime, execution_time, False))

        if is_runtime:
            device_loads[device_id] += 100 * 5
        else:
            device_loads[device_id] += execution_time
    return jobs


def least_busy(is_runtime_all, circuit_times, devices):
    jobs = []
    device_loads = [0] * len(devices)

    for i, (is_runtime, execution_time) in enumerate(
        zip(is_runtime_all, circuit_times)
    ):
        min_load = min(device_loads)
        candidates = [i for i, load in enumerate(device_loads) if load == min_load]
        device_id = candidates[i % len(candidates)]

        jobs.append(generate_job(i, device_id, is_runtime, execution_time, False))

        if is_runtime:
            device_loads[device_id] += 100 * 5
        else:
            device_loads[device_id] += execution_time

    return jobs


def fidelity_weighted(is_runtime_all, circuit_times, devices):
    fidelities = [device.fidelity for device in devices]
    normalized_fidelities = np.array(fidelities) / sum(fidelities)

    jobs = []

    for i, (is_runtime, execution_time) in enumerate(
        zip(is_runtime_all, circuit_times)
    ):
        device_id = np.random.choice(len(devices), p=normalized_fidelities)

        jobs.append(generate_job(i, device_id, is_runtime, execution_time, False))
    return jobs


def best_fidelity(is_runtime_all, circuit_times, devices):
    jobs = []
    device_loads = [0] * len(devices)

    fidelities = [device.fidelity for device in devices]
    normalized_fidelities = np.array(fidelities) / sum(fidelities)

    for i, (is_runtime, execution_time) in enumerate(
        zip(is_runtime_all, circuit_times)
    ):
        if is_runtime:
            high_fideilty_loads = device_loads[:3]
            min_subset_load = min(high_fideilty_loads)
            min_subset_candidates = [
                i
                for i, load in enumerate(high_fideilty_loads)
                if load == min_subset_load
            ]
            device_id = min_subset_candidates[i % len(min_subset_candidates)]

        else:

            device_id = np.random.choice(len(devices), p=normalized_fidelities)

        jobs.append(generate_job(i, device_id, is_runtime, execution_time, False))

        if is_runtime:
            device_loads[device_id] += 100 * 5
        else:
            device_loads[device_id] += execution_time
    return jobs


def qoncord(is_runtime_all, circuit_times, devices):
    jobs = []
    device_loads = [0] * len(devices)

    for i, (is_runtime, execution_time) in enumerate(
        zip(is_runtime_all, circuit_times)
    ):
        if is_runtime:
            subset = [6, 7, 8]

            # Find the device with the minimum load overall
            min_load_device = device_loads.index(min(device_loads))

            # Find the minimum load within the subset
            subset_loads = [device_loads[i] for i in subset]
            min_subset_load = min(subset_loads)
            device_id = subset[subset_loads.index(min_subset_load)]

        else:
            min_load = min(device_loads)
            candidates = [i for i, load in enumerate(device_loads) if load == min_load]
            device_id = candidates[i % len(candidates)]

        jobs.append(generate_job(i, device_id, is_runtime, execution_time, True))

        if is_runtime:
            device_loads[device_id] += 100 * 5
        else:
            device_loads[device_id] += execution_time
    return jobs


def eqc_scheduling(is_runtime_all, circuit_times, devices):
    jobs = []
    device_loads = [0] * len(devices)

    job_index = 0
    total_job_time = 0
    for is_runtime, execution_time in zip(is_runtime_all, circuit_times):
        if is_runtime:
            n_circuits = 100 * 2
            execution_time = 5
        else:
            n_circuits = 1

        for _ in range(n_circuits):
            min_load = min(device_loads)
            candidates = [i for i, load in enumerate(device_loads) if load == min_load]
            device_id = candidates[job_index % len(candidates)]

            jobs.append(
                generate_job(job_index, device_id, False, execution_time, False)
            )
            job_index += 1
            device_loads[device_id] += execution_time + 1
            total_job_time += execution_time

    return device_loads, total_job_time
