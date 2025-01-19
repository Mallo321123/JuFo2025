import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy.optimize import curve_fit

def quadratic(x,a,b,c):
    return a * x**2 + b * x +c

def plot_fitness_evolution(avg_f, best_f, genl):

    rcParams['font.family'] = 'DejaVu Serif'
    rcParams['font.size'] = 20

    gen = np.array(list(range(1,genl+1)))

    avg_params, _ = curve_fit(quadratic, gen, avg_f)
    best_params, _ = curve_fit(quadratic, gen, best_f)

    x_smooth = np.linspace(gen.min(), gen.max(), 500)
    avg_fit_curve = quadratic(x_smooth, *avg_params)
    best_fit_curve = quadratic(x_smooth, *best_params)

    fig, ax = plt.subplots(2,1, figsize=(10,8), sharex=True)

    ax[0].plot(gen, avg_f, marker='o', linestyle='-', color='blue', label='Average fitness')
    ax[0].plot(x_smooth, avg_fit_curve, '-', label='Fitted curve', color='cyan')
    ax[0].set_title('Average fitness across generations')
    ax[0].set_ylabel('Fitness')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(gen, best_f, marker='o', linestyle='-', color='green', label='Best fitness')
    ax[1].plot(x_smooth, best_fit_curve, '-', label='Fitted curve', color='lime')
    ax[1].set_title('Best fitness across generations')
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel('Fitness')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()

    plt.show()