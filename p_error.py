import qiskit
import numpy as np

def get_p(t1, t2, t_mu, cd, g1_error, g1, g2_error, g2, m_error, m):
    return (
        np.exp((-1 * t_mu * cd) / (t1 * t2))
        * (1 - g1_error) ** g1
        * (1 - g2_error) ** g2
        * (1 - m_error) ** m
    )


def calculate_p_err(circuit, backend, transpiled=False):

    sq_names = ["ID", "RZ", "SX", "X"]
    cx_names = ["CX"]

    if transpiled:
        t_circs = circuit
    else:
        t_circs = qiskit.transpile(
            circuit, backend=backend, routing_method="sabre", optimization_level=3
        )

    t_sq = 0
    t_cx = 0

    for key in t_circs.count_ops().keys():
        if key.upper() in sq_names:
            t_sq += t_circs.count_ops()[key]
        elif key.upper() in cx_names:
            t_cx += t_circs.count_ops()[key]
        else:
            pass
    data = backend.properties().to_dict()

    # Collects list of CNOT gate time and SQG gate time
    cx_times = []
    sq_times = []
    cx_error = []
    sq_error = []
    for gate_info in data["gates"]:
        if len(gate_info["parameters"]) < 2:
            continue
        if gate_info["gate"] == "cx":
            cx_times.append(gate_info["parameters"][1]["value"])
            cx_error.append(gate_info["parameters"][0]["value"])
        else:
            time = gate_info["parameters"][1]["value"]
            sq_times.append(time)
            sq_error.append(gate_info["parameters"][0]["value"])
    t1_times = []
    m_error = []
    t2_times = []
    for qubit in data["qubits"]:
        t1_times.append(qubit[0]["value"])
        t2_times.append(qubit[1]["value"])
        m_error.append(qubit[4]["value"])
    t1_times = np.array(t1_times)
    t2_times = np.array(t2_times)
    sq_times = np.array(sq_times)
    cx_times = np.array(cx_times)
    m_error = np.array(m_error)
    cx_error = np.array(cx_error)
    sq_error = np.array(sq_error)
    t1 = np.mean(t1_times) * 1000  # Unit conversion
    t2 = np.mean(t2_times) * 1000  # Unit conversion
    sq = np.mean(sq_error)
    cx = np.mean(cx_error)
    m = np.mean(m_error)
    transpiled_circ = qiskit.transpile(circuit, backend=backend)
    depth = transpiled_circ.depth()
    t_mu = (cx_times.mean() + sq_times.mean()) / 2
    return get_p(
        t1, t2, t_mu, depth, sq, t_sq, cx, t_cx, m, t_circs.count_ops()["measure"]
    )
