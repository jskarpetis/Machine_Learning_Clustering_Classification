import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

def gaussian_kernel(x,h):
    return (1 / (np.sqrt(2 * np.pi * h**2)) * np.exp((-1 / (2 * h))*x**2))


def show_plots(plots):
    counter=0
    plt.tight_layout()

    for h in plots:
        counter+=1
        plt.subplot(4,1,counter)
        plt.plot(X_plot, plots[h], '.')
        plt.grid(1)

    plt.show(block=True)


if __name__ == "__main__":
        
    h_list = [0.004, 0.008, 0.01, 0.02]
    
    final_results = {}
    
    X = uniform.rvs(size=1000, loc=0, scale=1)
    N=len(X) 

    X_plot = np.linspace(-3, 3, 1000)[:, None]    
    X_plot = np.reshape(X_plot, (1000,))

    for h in h_list:
        summary = 0
        for i in range(N):
            summary += gaussian_kernel(X_plot - X[i], h) / N
        final_results[h] = summary
            
    show_plots(final_results)
    
    

        