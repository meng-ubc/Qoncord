import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams.update({'font.size': 8,
                     'figure.dpi': 300})

def plot_multi_restarts(data, filename=None):
    
    if data is None:
        with open(filename, 'r') as f:
            data = json.load(f)

    total_approx_vals = data['approx']
    nfev_data = data['nfev']

    # Create an empty DataFrame
    approx_data = pd.DataFrame(columns=['Category', 'Approximation Ratio'])

    labels = ['Qoncord', 'HF', 'LF']
    # Populate the DataFrame
    for label, vals in zip(labels, total_approx_vals):
        temp_df = pd.DataFrame({'Category': label, 'Approximation Ratio': vals})
        approx_data = pd.concat([approx_data, temp_df], ignore_index=True)
       
    fig, (ax1, ax2)= plt.subplots(2, 1, figsize=(3, 3.5))

    order = ['LF', 'HF', 'Qoncord']

    sns.boxplot(x='Category', y='Approximation Ratio', data=approx_data, linewidth=1, order=order, fliersize=5, color='black', ax=ax1)

    # Customize median line color and line width
    median_color = 'red'
    median_linewidth = 1.5
    for line in ax1.lines[4::6]:  # Median lines are every 6th line starting from the 5th
        line.set_color(median_color)
        line.set_linewidth(median_linewidth)
        
    ax1.set_ylabel('Approximation Ratio\n(Higher is Better)')
    ax1.grid(axis='y', linestyle='--')
    ax1.set_xlabel('')

    qoncord_hf_nfev = sum(nfev_data[0])
    qoncord_lf_nfev = sum(nfev_data[1])
    hf_nfev = sum(nfev_data[2])
    lf_nfev = sum(nfev_data[3])

    ax2.bar(0.4, lf_nfev, color="#9BBEC8" , width=0.5,zorder=3, label='LF Device')
    ax2.bar(1, hf_nfev, color="#164863", width=0.5,zorder=3, label='HF Device')
    ax2.bar(1.6, qoncord_lf_nfev, width=0.5, color="#9BBEC8" , zorder=3)
    ax2.bar(1.6, qoncord_hf_nfev, bottom=qoncord_lf_nfev, width=0.5, color="#164863", zorder=3)

    ax2.set_ylabel('Circuit Execution Overhead\n(Lower is Better)')
    ax2.set_xticks([0.4, 1, 1.6], ['LF', 'HF','Qoncord'])
    ax2.grid(axis='y', linestyle='--', zorder=0)
    ax2.tick_params(axis='x')
    ax2.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3))

    plt.tight_layout()
    plt.show()
    
    
def plot_single_restart(data, filename=None):
    
    if data is None:
        with open(filename, 'r') as f:
            data = json.load(f)
    c1 = '#DDF2FD'
    c2 = '#164863'
    c3 = '#9BBEC8'

    opt_vals = data['approx']
    nits = data['nfev']
    
    width = 0.8
    x_values = np.arange(3)

    labels = ['LF', 'HF', 'Qoncord']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.375, 1.7), sharey=False)

    mean_opt_vals = np.mean(opt_vals, axis=1)
    std_opt_vals = np.std(opt_vals, axis=1)

    mean_nits = np.mean(nits, axis=1)
    std_nits = np.std(nits, axis=1)

    ax1.bar(x_values[0], mean_opt_vals[2], width, yerr=std_opt_vals[2], capsize=3, color=c3, zorder=3)
    ax1.bar(x_values[1], mean_opt_vals[1], width, yerr=std_opt_vals[1], capsize=3, color=c2, zorder=3)
    ax1.bar(x_values[2], mean_opt_vals[0], width, yerr=std_opt_vals[0], capsize=3, color=c1, zorder=3)


    ax1.set_ylim(0.6, 0.85)
    ax1.grid(axis='y', linestyle='--', linewidth=1, zorder=0)
    ax1.set_xticks(x_values, labels)
    ax1.set_ylabel('Approximation Ratio')

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax2.bar(x_values[0], mean_nits[3], width, yerr=std_nits[3], capsize=3, color=c3, zorder=3, label='LF')
    ax2.bar(x_values[1], mean_nits[2], width, yerr=std_nits[2], capsize=3, color=c2, zorder=3, label='HF')
    # ax2.bar(x_values[2], mean_nits[0], width, yerr=std_nits[2], capsize=3, color=c1, zorder=3)
    ax2.bar(x_values[2], mean_nits[1], width, color=c3, zorder=3)
    ax2.bar(x_values[2], mean_nits[0], width, bottom=mean_nits[1], yerr=std_nits[1] + std_nits[0], capsize=3,  color=c2, zorder=3)
    ax2.grid(axis='y', linestyle='--', linewidth=1, zorder=0)
    ax2.set_xticks(x_values, labels)
    ax2.set_ylabel('Number of Circuit Evaluations')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2, columnspacing=0.3)
    ax2.set_ylim(0, 130)

    plt.tight_layout()
    plt.show()
    
def plot_async_results(data=None):
    if data is None:
        with open('async_results.json', 'r') as f:
            data = json.load(f)
    
    # Plotting the results
    fig, ax1 = plt.subplots(figsize=(3.375, 2.5))

    # Create the secondary axis
    ax2 = ax1.twinx()

    # Data for the grouped bar chart
    n_groups = 2
    index = np.arange(n_groups)

    bar_width = 0.35

    opt_index = [index[0] - 0.005, index[0] + bar_width + 0.005]
    nfev_index = [index[1]- 0.005, index[1] + bar_width + 0.005]

    baseline_color = '#164863'
    async_color = '#9BBEC8'

    # Plotting
    ax1.bar(opt_index[0], np.mean(data['baseline_opt']), bar_width, color=baseline_color,  label='Baseline',zorder=3)
    ax1.bar(opt_index[1], np.mean(data['async_opt']), bar_width, color=async_color,  label='Async (EQC)',zorder=3)
    ax1.errorbar(opt_index[0], np.mean(data['baseline_opt']), yerr=np.std(data['baseline_opt']), color='black', capsize=5,zorder=4)
    ax1.errorbar(opt_index[1], np.mean(data['async_opt']), yerr=np.std(data['async_opt']), color='black', capsize=5,zorder=4)

    ax2.bar(nfev_index[0], np.mean(data['baseline_nfev']), bar_width,color=baseline_color,  zorder=3)
    ax2.bar(nfev_index[1], np.mean(data['async_nfev']), bar_width,color=async_color,  zorder=3)
    ax2.errorbar(nfev_index[0], np.mean(data['baseline_nfev']), yerr=np.std(data['baseline_nfev']), color='black', capsize=5,zorder=4)
    ax2.errorbar(nfev_index[1], np.mean(data['async_nfev']), yerr=np.std(data['async_nfev']), color='black', capsize=5,zorder=4)

    # Labeling
    ax1.set_ylabel('Approximation Ratio')
    ax2.set_ylabel('Number of Circuit Executions')

    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(['Approximation Ratio', 'nfev'])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    
    ax1.set_ylim(0.7, 0.95)
    
    # Get raw y limits of both axes
    ylim_ax1 = ax1.get_ylim()
    ylim_ax2 = ax2.get_ylim()

    # Set a fixed number of ticks
    n_ticks = 5

    # Generate ticks for ax1 and ax2
    ticks_ax1 = np.linspace(ylim_ax1[0], ylim_ax1[1], n_ticks)
    ticks_ax2 = np.linspace(ylim_ax2[0], ylim_ax2[1], n_ticks)

    # Set the ticks for both axes
    ax1.set_yticks(ticks_ax1)
    ax2.set_yticks(ticks_ax2)

    # Get current tick locations
    ticks_loc_ax1 = ax1.get_yticks()
    ticks_loc_ax2 = ax2.get_yticks()

    # Round the tick labels: two decimal places for ax1, integers for ax2
    labels_ax1 = [f"{x:.2f}" for x in ticks_loc_ax1]
    labels_ax2 = [f"{int(x)}" for x in ticks_loc_ax2]

    # Set new tick labels
    ax1.set_yticklabels(labels_ax1)
    ax2.set_yticklabels(labels_ax2)

    # Adding a grid to ax1
    ax1.grid(True, zorder=0, axis='y')

    ax1.tick_params(axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)

    ax1.set_xticklabels(['Result Quality', 'Execution Overhead'])
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_results.py <experiment_index>")
        exit(0)
    
    index = int(sys.argv[1])
    
    if index == 2:
        plot_multi_restarts(None, 'qaoa_results.json')
    elif index == 3:
        plot_single_restart(None, 'vqe_results.json')
    elif index == 4:
        plot_async_results(None)
    else:
        print("Invalid experiment number")