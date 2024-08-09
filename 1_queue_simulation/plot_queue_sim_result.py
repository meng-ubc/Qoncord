import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_queue_sim(data):

    number_of_circuits = data[data['policy'] == "random"]['total_circuits'].tolist()
    number_of_circuits_eqc = data[data['policy'] == "eqc_scheduling"]['total_circuits'].tolist()

    # Create a dictionary to map runtime_ratio (as string) to number of circuits
    ratio_to_circuit_dict = {f"{0.1 * i:.1f}": number_of_circuits[i-1] for i in range(1, 10)}
    ratio_to_circuit_dict_eqc = {f"{0.1 * i:.1f}": number_of_circuits_eqc[i-1] for i in range(1, 10)}

    # Select only the columns you're interested in
    filtered_df = data[['runtime_ratio', 'policy', 'max_device_completion_time', 'average_fidelity']].copy()
    # Calculate 'throughput' using the appropriate dictionary based on policy
    filtered_df['throughput'] = filtered_df.apply(
        lambda row: (ratio_to_circuit_dict_eqc[f"{row['runtime_ratio']:.1f}"]
            if row['policy'] == 'eqc_scheduling'
            else ratio_to_circuit_dict[f"{row['runtime_ratio']:.1f}"]) / row['max_device_completion_time'],
        axis=1
    )

    min_value = filtered_df['max_device_completion_time'].min()
    max_value = filtered_df['max_device_completion_time'].max()

    # Calculate the normalized 'max_device_completion_time'
    filtered_df['normalized_max_device_completion_time'] = filtered_df['max_device_completion_time'] / min_value
    filtered_df['speedup'] = max_value / filtered_df['max_device_completion_time']

    max_fidelity = filtered_df['average_fidelity'].max()
    filtered_df['normalized_average_fidelity'] = filtered_df['average_fidelity'] / max_fidelity

    # Replace this list with the policies you wish to exclude
    policies_to_discard = ['random']

    # Filter out the unwanted policies
    filtered_df = filtered_df[~filtered_df['policy'].isin(policies_to_discard)]

    filtered_df = filtered_df.drop(filtered_df[(filtered_df.runtime_ratio == 0.2) | 
                                            (filtered_df.runtime_ratio == 0.4) |
                                            (filtered_df.runtime_ratio == 0.6) | 
                                            (filtered_df.runtime_ratio == 0.8)].index)


    plt.rcParams.update({'font.size': 9,
                        'figure.dpi': 150})

    markers = ['D', 'v', 's', 'P', "^", 'o']
    # Create a scatter plot
    plt.figure(figsize=(8, 3))
    plt.title('Current Device Fidelity')
    scatter = sns.scatterplot(data=filtered_df, 
                    x='throughput', # normalized_max_device_completion_time 
                    y='normalized_average_fidelity', 
                    hue='runtime_ratio', 
                    style='policy',
                    palette='viridis',
                    markers = markers,
                    legend=True,
                    s=50,
                    zorder=3)  # Using a grayscale palette

    # Extract the handles and labels
    handles, labels = scatter.get_legend_handles_labels()

    # Define new order and labels for the 'policy' part of the legend
    # Adjust this according to the actual policies in your DataFrame
    new_order = ['least_busy', 'random_load_weighted',  'fidelity_weighted', 'best_fidelity', 'eqc_scheduling', 'qoncord']  # New order
    new_labels = ['Least Busy', 'Load Weighted', 'Fidelity Weighted', 'Best Fideilty', 'EQC', 'Qoncord']  # New labels

    # Reorder and relabel handles and labels
    new_handles = [handles[labels.index(policy)] for policy in new_order if policy in labels]
    new_labels = [new_labels[new_order.index(policy)] for policy in new_order if policy in labels]

    # Add runtime ratio part of the legend (assuming it's the first part)
    runtime_ratio_handles = handles[:len(set(filtered_df['runtime_ratio'])) + 1]
    runtime_ratio_labels = labels[:len(set(filtered_df['runtime_ratio'])) + 1]
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
        
    runtime_ratio_labels = [f'{float(label):.1f}' if  is_float(label) else label for label in runtime_ratio_labels]

    # Combine both parts of the legend
    final_handles = runtime_ratio_handles + new_handles
    final_labels = runtime_ratio_labels + new_labels
    final_labels[0] = 'Runtime Ratio'

    # Adding customized legend
    plt.legend(final_handles, final_labels, loc='right', bbox_to_anchor=(1.8, 0.5), title_fontsize='small')

    plt.grid(axis='both',  linestyle='--', alpha=0.7, zorder=0)
    # Adding labels and title
    plt.xlabel('Throughput (Circuits/Unit Time)')
    plt.ylabel('Execution Fidelity')

    plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    data = pd.read_csv('queue_sim_results.csv')
    plot_queue_sim(data)