import numpy as np
import networkx as nx

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

import math


def maxcut_obj(x, G):
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1

    return obj


def compute_expectation(counts, G):
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = maxcut_obj(bitstring[::-1], G)
        avg += obj * count
        sum_count += count

    return avg / sum_count


def create_qaoa_circ(theta, G):
    nqubits = len(G.nodes())
    p = len(theta) // 2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)

    beta = theta[:p]
    gamma = theta[p:]

    # initial_state
    for i in range(0, nqubits):
        qc.h(i)

    for irep in range(0, p):
        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(gamma[irep], pair[0], pair[1])

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(beta[irep], i)

    qc.measure_all()

    return qc


def filter_counts(counts, percentage=0.5):
    # Get the total count
    total = sum(counts.values())

    # Get 90% of total
    ninety_percent = percentage * total

    # Sort the outcomes by frequency
    sorted_outcomes = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Initialize count
    count = 0

    total_outcomes = {}

    # Loop through outcomes in descending order
    for outcome, freq in sorted_outcomes:
        # Add the frequency
        count += freq

        total_outcomes[outcome] = freq

        # If count exceeds 90% of total, break
        if count >= ninety_percent:
            break

    return total_outcomes


def counts_to_statevector(counts):
    num_qubits = len(list(counts.keys())[0])

    # Initialize a state vector with zeros
    state_vector = np.zeros(2**num_qubits, dtype=complex)

    # Total number of shots
    total_shots = sum(counts.values())

    # Assign amplitudes to the state vector based on counts
    for state, count in counts.items():
        index = int(state, 2)  # Convert binary string to integer
        probability = count / total_shots  # Calculate probability
        amplitude = np.sqrt(probability)  # Assign amplitude
        state_vector[index] = amplitude

    sv = Statevector(state_vector)
    assert sv.is_valid(), "Statevector is not valid"
    return sv


def get_expectation(
    G,
    p,
    backend=None,
    noise_model=None,
    shots=8192,
    GPU=False
):

    parameters = [Parameter("theta" + str(i)) for i in range(2 * p)]

    circuit = create_qaoa_circ(parameters, G)

    if backend is not None:
        # Setup noisy simulation
        aer_sim = AerSimulator.from_backend(backend)
        aer_sim.set_options(method="density_matrix")

        circuit = transpile(
            circuit, backend=backend, routing_method="sabre", optimization_level=3
        )
    elif noise_model is not None:
        aer_sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        circuit = transpile(
            circuit, basis_gates=noise_model.basis_gates, optimization_level=3
        )
    else:
        aer_sim = AerSimulator()

    if GPU:
        aer_sim.set_options(device="GPU")

    def execute_circ(
        theta
    ):
        theta = np.array(theta)
        assert len(theta) == 2 * p, "theta must be of length 2p"

        sim_circuit = circuit.assign_parameters(
            {parameters[i]: theta[i] for i in range(2 * p)}
        )
        sim_shots = shots

        result = aer_sim.run(sim_circuit, shots=sim_shots).result()

        counts = result.get_counts()
        return compute_expectation(counts, G)


    return execute_circ
