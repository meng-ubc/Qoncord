from qaoa_util import get_expectation
import networkx as nx
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
from plot_results import plot_async_results
import json

plt.rcParams.update({'font.size': 8,
                     'figure.dpi': 150})

def async_exp_fun_wrapper(fun, free_index, base_params):
    
    def wrapper(params):
        base_params[free_index] = params[0]
        
        return fun(base_params)
    
    return wrapper

def async_optimization(exp_fun, init_param, num_repeats=10):
    async_opt = []
    async_nfev = []

    for _ in tqdm(range(num_repeats), desc='Async (EQC)'):
        current_params = init_param.copy()
        opt_params = []
        
        fun_values = []
        nfev_values = []

        for i in range(len(init_param)):
            # Create a wrapper function for the current parameter
            async_exp_fixed = async_exp_fun_wrapper(exp_fun, i, current_params)

            # Optimize the current parameter
            res = minimize(async_exp_fixed, current_params[i], method='COBYLA')
            
            # Update the parameter in the list
            opt_params.append(res.x[0])

            # Collect function values and number of function evaluations
            fun_values.append(res.fun)
            nfev_values.append(res.nfev)
            
        fun_values.append(exp_fun(opt_params))
        
        async_opt.append(fun_values)
        async_nfev.append(nfev_values)
        # print(fun_values[-1], sum(nfev_values))

    return async_opt, async_nfev

# Get the benchmarks
g = nx.gnp_random_graph(10, 0.5, seed=42)
p = 3
maxcut = nx.algorithms.approximation.one_exchange(g)[0]
exp_fun = get_expectation(g, p)

init_param = np.random.uniform(0, 2*np.pi, 2 * p)

# Get basline results where all parameters are optimized at once
baseline_opt = []
baseline_nfev = []
for _ in tqdm(range(10), desc='Baseline'):
    res = minimize(exp_fun, init_param, method='COBYLA')
    baseline_opt.append(res.fun)
    baseline_nfev.append(res.nfev)
    # print(res.fun, res.nfev)

# Get async results where each parameter is optimized separately
async_opt, async_nfev = async_optimization(exp_fun, init_param, num_repeats=10)

# Get the approximation ratio and number of function evaluations
baseline_opt = [opt / -maxcut for opt in baseline_opt]
async_opt = [opt[-1] / -maxcut for opt in async_opt]
async_nfev = [sum(nfev) for nfev in async_nfev]

data = {
    'baseline_opt': baseline_opt,
    'async_opt': async_opt,
    'baseline_nfev': baseline_nfev,
    'async_nfev': async_nfev
}


with open('async_results.json', 'w') as f:
    json.dump(data, f)
    
    
# plot_async_results(data)