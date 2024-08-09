import numpy as np
import networkx as nx

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

def get_h2_hamiltonian():
    return SparsePauliOp.from_list(
        [
            ("IIII", -1.0523732),
            ("ZIII", 0.39793742),
            ("IZII", -0.3979374),
            ("ZZII", -0.0112801),
            ("IIZI", 0.3979374),
            ("IIIZ", -0.3979374),
            ("IIZZ", -0.0112801),
            ("ZIIZ", 0.18093119),
            ("IZZI", -0.18093119),
            ("ZIZI", 0.18105829),
            ("XXXX", 0.045322202),
            ("YYXX", 0.045322202),
            ("XXYY", 0.045322202),
            ("YYYY", 0.045322202),
        ]
    )


def create_uccsd_ansatz():
    # Assuming num_qubits = 4 for the hydrogen molecule
    # 6 parameters for single excitations and 3 for double excitations
    num_qubits = 4
    parameters = []

    qc = QuantumCircuit(num_qubits)

    # Single excitations
    param_idx = 0
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            parameters.append(Parameter("theta" + str(param_idx)))

            qc.rx(parameters[-1], i)
            qc.cx(i, j)
            qc.rz(parameters[-1], j)
            qc.cx(i, j)
            qc.rx(-parameters[-1], i)
            param_idx += 1

    # Double excitations (simplified version)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            for k in range(j + 1, num_qubits):
                for l in range(k + 1, num_qubits):
                    # Double excitation circuit segment

                    parameters.append(Parameter("theta" + str(param_idx)))
                    qc.cx(i, j)
                    qc.cx(k, l)
                    qc.rz(parameters[-1], l)
                    qc.cx(k, l)
                    qc.cx(i, j)
                    param_idx += 1
    qc.measure_all()
    return qc, parameters

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

def get_expectation_vqe(
    backend=None,
    noise_model=None,
    shots=8192,
    GPU=False,
):
    
    circuit, parameters = create_uccsd_ansatz()
    num_params = len(parameters)
    observable = get_h2_hamiltonian()

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

    def execute_circ(theta):
        theta = np.array(theta)
        assert len(theta) == num_params, f"theta must be of length {num_params}"

        sim_circuit = circuit.assign_parameters(
            {parameters[i]: theta[i] for i in range(num_params)}
        )
        sim_shots = shots

        result = aer_sim.run(sim_circuit, shots=sim_shots).result()

        counts = result.get_counts()

        sv = counts_to_statevector(counts)
        return sv.expectation_value(observable).real

    return execute_circ
