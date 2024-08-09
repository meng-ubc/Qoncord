# Qoncord Experiment Reproduction

This repository contains the scripts necessary to reproduce the key results presented in our paper: *Qoncord: A Multi-Device Job Scheduling Framework for Variational Quantum Algorithms*. Each experiment has its dedicated Python script for execution, and a separate script for generating the corresponding plots.

## Table of Contents

- Prerequisites
- Setup
- Experiments
  - Queue Simulation
  - Multi-Restart QAOA Optimization
  - Single-Restart VQE Optimization
  - Comparison to Asynchronous Gradient Descent
- License

## Prerequisites

- Conda (for environment management)
- Python 3.11+

## Setup

1. **Clone the repository:**
   
   git clone https://github.com/meng-ubc/Qoncord.git  
   cd Qoncord

2. **Create and activate the Conda environment:**
   
   conda create -n qoncord_env python=3.11  
   conda activate qoncord_env

3. **Install the required packages:**
   
   pip install -r requirements.txt

## Experiments

Each experiment has its own script. Running these scripts will execute the experiment and save the results for plotting.

### Queue Simulation

- **Script:** main_queue_sim.py
- **Expected Runtime:** Approximately 2 minutes
- **Output:** queue_sim_results.csv

To run the experiment:
    
    cd 1_queue_simulation
    python queue_simulation.py

To plot the result fiugre:
    
    python plot_queue_sim_result.py

### Multi-Restart QAOA Optimization

- **Script:** 2_qaoa_optimization.py
- **Expected Runtime:** Approximately 10 minutes
- **Output:** qaoa_results.json

To run the experiment:
   
    python 2_qaoa_optimization.py

To plot the result fiugre:
    
    python plot_results.py 2

### Single-Restart VQE Optimization

- **Script:** 3_vqe_optimization.py
- **Expected Runtime:** Approximately 4 minutes
- **Output:** vqe_results.json

To run the experiment:

    python 3_vqe_optimization.py

To plot the result fiugre:
    
    python plot_results.py 3

### Comparison to Asynchronous Gradient Descent

- **Script:** 4_ae_async.py
- **Expected Runtime:** Approximately 6 minutes
- **Output:** async_results.json

To run the experiment:
   
    python 4_ae_async.py

To plot the result fiugre:
    
    python plot_results.py 4

## License

This project is licensed under the MIT License. See the LICENSE file for details.
