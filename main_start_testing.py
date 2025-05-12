# filename: main_start_testing.py
# Run this file to test the diffrent algorithms

import numpy as np
import random
import pickle
from tqdm import tqdm
from neural_network import NeuralNetwork
from fitness_function import fitness_score
import matplotlib.pyplot as plt
import json
from traffic_simulation import (
    execute_simulation_nn,
    execute_simulation_algt,
    execute_simulation_algs,
)

fitness_ENN = []
fitness_t_plus = []
fitness_timed = []

fittness_ENN_total = []
fittness_t_plus_total = []
fittness_timed_total = []

TRAFFIC_A = 0.27

# Imports a saved model from the saved_model.pickle file.
def get_saved_model():
    with open(
        "test1.pickle", "rb"
    ) as file:
        neural_network = pickle.load(file)
    return neural_network


def start_testing(neural_network, n):
    # Setting how many cars appear per timestep on average.
    # Because this a testing enviorment the value should remain a constant for all the algorithms.
    # Traffic and disapering traffic are varied by the poisson function for realism.
    def def_traffic_at_A():
        # training_values
        #traffic_at_A = random.uniform(0.1, 0.3)
        traffic_at_A = n
        return traffic_at_A

    def def_traffic_at_B():
        # training_values
        # traffic_at_B = random.uniform(0.1, 0.3)
        traffic_at_B = 0.3 - traffic_at_A
        return traffic_at_B

    traffic_at_A = def_traffic_at_A()
    traffic_at_B = def_traffic_at_B()
    
    # Setting the duration of the testing simulation
    def def_steps():
        steps = 500
        return steps

    def def_disapearing_traffic():
        d_traffic = 0.2
        return d_traffic

    # Testing the neural_network
    fitness_nn = execute_simulation_nn(
        traffic_at_A,
        traffic_at_B,
        def_disapearing_traffic(),
        def_disapearing_traffic(),
        def_steps(),
        neural_network,
        show=False,
    )
    print("Fitness:", fitness_nn)
    print("\n")
    fitness_ENN.append(fitness_nn)

    # Controll: regular timed algorithm
    fitness_algt = execute_simulation_algt(
        traffic_at_A,
        traffic_at_B,
        def_disapearing_traffic(),
        def_disapearing_traffic(),
        def_steps(),
        show=False,
    )
    print("Fitness:", fitness_algt)
    print("\n")
    fitness_t_plus.append(fitness_algt)

    # Comparission: regular timed algorithm wirh switching after one side has cleared
    fitness_algs = execute_simulation_algs(
        traffic_at_A,
        traffic_at_B,
        def_disapearing_traffic(),
        def_disapearing_traffic(),
        def_steps(),
        show=False,
    )
    print("Fitness:", fitness_algs)
    print("\n")
    fitness_timed.append(fitness_algs)


neural_network = get_saved_model()

# Lists to store standard deviations for each traffic value
std_ENN_total = []
std_t_plus_total = []
std_timed_total = []

for n in tqdm(range(0, 31, 1)):
    # Clear lists for each traffic value
    fitness_ENN = []
    fitness_t_plus = []
    fitness_timed = []
    
    for i in range(50):
        start_testing(neural_network, n/100)
    
    print(f"traffic_at_A:{n/100}")
    print("ENN_std:", np.std(fitness_ENN))
    print("ENN_mean:", np.mean(fitness_ENN))
    print("plus_std:", np.std(fitness_t_plus))
    print("plus_mean:", np.mean(fitness_t_plus))
    print("Timed+std:", np.std(fitness_timed))
    print("Timed+mean:", np.mean(fitness_timed))
    
    # Store means and standard deviations
    fittness_ENN_total.append(np.mean(fitness_ENN))
    fittness_t_plus_total.append(np.mean(fitness_t_plus))
    fittness_timed_total.append(np.mean(fitness_timed))
    
    std_ENN_total.append(np.std(fitness_ENN))
    std_t_plus_total.append(np.std(fitness_t_plus))
    std_timed_total.append(np.std(fitness_timed))
    
print("ENN_total_std:", np.std(fittness_ENN_total))
print("ENN_total_mean:", np.mean(fittness_ENN_total))
print("plus_total_std:", np.std(fittness_t_plus_total))
print("plus_total_mean:", np.mean(fittness_t_plus_total))
print("Timed+total_std:", np.std(fittness_timed_total))
print("Timed+total_mean:", np.mean(fittness_timed_total))

# Create dictionary with all results
results = {
    "traffic_at_A": TRAFFIC_A,
    "neural_network": {
        "fitness_values": fitness_ENN,
        "mean": np.mean(fitness_ENN),
        "std": np.std(fitness_ENN),
        "total_mean": np.mean(fittness_ENN_total),
        "total_std": np.std(fittness_ENN_total)
    },
    "timed_plus": {
        "fitness_values": fitness_t_plus,
        "mean": np.mean(fitness_t_plus),
        "std": np.std(fitness_t_plus),
        "total_mean": np.mean(fittness_t_plus_total),
        "total_std": np.std(fittness_t_plus_total)
    },
    "timed": {
        "fitness_values": fitness_timed,
        "mean": np.mean(fitness_timed),
        "std": np.std(fitness_timed),
        "total_mean": np.mean(fittness_timed_total),
        "total_std": np.std(fittness_timed_total)
    }
}

# Prepare data for plotting
traffic_values = [i/100/0.3 for i in range(0, 31, 1)]

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot fitness means for each algorithm with error bars
ax1.errorbar(traffic_values, fittness_ENN_total, yerr=std_ENN_total, fmt='b-', label='Neural Network', capsize=3, color='blue')
ax1.errorbar(traffic_values, fittness_t_plus_total, yerr=std_t_plus_total, fmt='r-', label='Timed Plus', capsize=3, color='yellow')
ax1.errorbar(traffic_values, fittness_timed_total, yerr=std_timed_total, fmt='g-', label='Timed', capsize=3, color='grey')
ax1.set_xlabel('Prozentuales Verkehraufsaufkommen auf Seite A')
ax1.set_ylabel('Durchschnittliche Fitness')
ax1.set_title('Vergleich der verschiedener Schaltungsalgorithmen')
ax1.legend()
ax1.grid(True)

# Calculate the difference between Neural Network and best conventional algorithm
diff_values = []
for i in range(len(fittness_ENN_total)):
    conventional_best = max(fittness_t_plus_total[i], fittness_timed_total[i])
    diff = fittness_ENN_total[i] - conventional_best
    diff_values.append(diff)

# Plot the difference
ax2.plot(traffic_values, diff_values, 'm-')
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.set_xlabel('Prozentuales Verkehraufsaufkommen auf Seite A')
ax2.set_ylabel('ENN Performanz')
ax2.set_title('leistung des ENN im Vergleich zu den besten klassischen Algorithmen')
ax2.grid(True)

plt.tight_layout()
plt.savefig(f'performance_comparison_traffic_{TRAFFIC_A}.png')
plt.show()

# Save to JSON file
with open(f'test_results_traffic_{TRAFFIC_A}.json', 'w') as file:
    json.dump(results, file, indent=4)

print(f"Results saved to test_results_traffic_{TRAFFIC_A}.json")