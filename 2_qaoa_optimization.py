import numpy as np
import networkx as nx
from scipy.optimize import minimize
from qaoa_util import get_expectation, create_qaoa_circ
from p_error import calculate_p_err
import json
from tqdm import tqdm

from qiskit_ibm_runtime.fake_provider import FakeKolkata, FakeToronto

backends = [FakeToronto(), FakeKolkata()]

# Generate the graph and noise models
G = nx.gnp_random_graph(7, 0.5, seed=42)
maxcut = nx.algorithms.approximation.one_exchange(G)[0]
p = 1
restarts = 50

init_params = np.random.uniform(0, 2*np.pi, (restarts, 2*p))

# Calculate the p_error for each backend
qaoa_circ = create_qaoa_circ(init_params[0], G)
p_errors = [calculate_p_err(qaoa_circ, backend) for backend in backends]

# assert no p_error is smaller than 0.1, otherwise the device is too noisy for the experiment
assert all(p_error >= 0.1 for p_error in p_errors)

# sort the backends by p_error
argsort_perr = np.argsort(p_errors)
sorted_backends = [backends[i] for i in argsort_perr]

print("Following backends are selected for the experiment:")
for i, backend in enumerate(sorted_backends):
    print(f'Backend {i+1}: {backend.name()} with p_error {p_errors[argsort_perr[i]]:.2f}')

exp_fun_lf = get_expectation(G, p, backend=sorted_backends[0])
exp_fun_hf = get_expectation(G, p, backend=sorted_backends[1])

# Perform optimization on low-fidelity device only
lf_results = {
    'approx': [],
    'nfev': []
}
for x0 in tqdm(init_params, desc='LF-Only'):
    lf_opt = minimize(exp_fun_lf, x0, method='COBYLA')
    lf_results['approx'].append(lf_opt.fun / -maxcut)
    lf_results['nfev'].append(lf_opt.nfev)

# Perform optimization on high-fidelity device only
hf_results = {
    'approx': [],
    'nfev': []
}
for x0 in tqdm(init_params, desc='HF-Only'):
    hf_opt = minimize(exp_fun_hf, x0, method='COBYLA')
    hf_results['approx'].append(hf_opt.fun / -maxcut)
    hf_results['nfev'].append(hf_opt.nfev)

# Qoncord optimization: initialize with low-fidelity results and fine-tune on high-fidelity device
qoncord_results = {
    'approx': [],
    'lf_nfev': [],
    'hf_nfev': []
}

qoncord_intermediate = {
    'approx': [],
    'x': []
}

# Qoncord exploration phase
for x0 in tqdm(init_params, desc='Qoncord Exploration'):
    lf_opt = minimize(exp_fun_lf, x0, method='COBYLA', tol=1e-1, options={'rhobeg': 1.0}) # Optimize with a higher tolerance and initial step size for exploration
    qoncord_intermediate['approx'].append(lf_opt.fun / -maxcut)
    qoncord_intermediate['x'].append(lf_opt.x)
    qoncord_results['lf_nfev'].append(lf_opt.nfev)

# Early termination - proceed with the best sets of result from the exploration phase
best_result = np.max(qoncord_intermediate['approx'])
continue_indicies = []

for i, result in enumerate(qoncord_intermediate['approx']):
    if result >= best_result - 0.02:
        continue_indicies.append(i)
    
    # Terminate early if we have more than half of the results
    if len(continue_indicies) >= len(qoncord_intermediate['approx']) / 2:
        break
        
print(f"Qoncord continuing with {len(continue_indicies)} sets of parameters to fine-tune on high-fidelity device")

# Qoncord fine-tuning phase
for i in tqdm(continue_indicies, desc='Qoncord Fine-Tuning'):
    hf_opt = minimize(exp_fun_hf, qoncord_intermediate['x'][i], method='COBYLA', options={'rhobeg': 0.1}) 
    
    qoncord_results['approx'].append(hf_opt.fun / -maxcut)
    qoncord_results['hf_nfev'].append(hf_opt.nfev)


total_opt_vals = [qoncord_results['approx'], hf_results['approx'], lf_results['approx']]
total_nfev_vals = [qoncord_results['hf_nfev'], qoncord_results['lf_nfev'], hf_results['nfev'], lf_results['nfev']]

with open('qaoa_results.json', 'w') as f:
    json.dump({'approx': total_opt_vals, 'nfev': total_nfev_vals}, f)