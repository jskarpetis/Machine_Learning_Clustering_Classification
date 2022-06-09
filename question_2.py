from mat4py import loadmat
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

def gaussian_kernel(X, Y, h):
    # return np.exp((-1 / h) * ((X[0] - Y[0])**2 + (X[1] - Y[1])**2))

    X = np.array(X)
    Y = np.array(Y)
    return np.exp((-1/h) * np.linalg.norm(X-Y)**2)
    

def load_data():
    data = loadmat('./Data/data32.mat')
    circle_data = data['circles']
    star_data = data['stars']
    
    return circle_data, star_data


def f_hat(X, Xi, Xj, ai, bj, h):
    s1 = sum([a*gaussian_kernel(X, xi, h) for xi, a in zip(Xi, ai)])
    s2 = sum([b*gaussian_kernel(X, xj, h) for xj, b in zip(Xj, bj)])
    return s1 + s2

def cost_function(Xi, Xj, X, l, h, ai, bj):
    
    sum_xi = 0
    for xi in Xi:
        sum_xi = sum_xi + (1 - f_hat(xi, Xi, Xj, ai, bj, h))**2
    
    sum_xj = 0
    for xj in Xj:
        sum_xj = sum_xj + (1 + f_hat(xj, Xi, Xj, ai, bj, h))**2        
    
    final_term = l * (np.linalg.norm(f_hat(X, Xi, Xj, ai, bj, h)))**2
    
    result = sum_xi + sum_xj + final_term
    
    # Goal is to minimize the result
    return result


def kernel_calculation(stars, circles, h):
    
    all_data = stars + circles
    n = len(all_data)
    
    
    labels = []
    for i in stars:
        labels.append(1)
    
    for j in circles:
        labels.append(-1)
    
    
    kernel = np.zeros((n,n))
    q = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            kernel[i,j] = gaussian_kernel(all_data[i], all_data[j], h)
            kernel[j,i] = kernel[i,j]
            q[i,j] = labels[i]*labels[j]*kernel[i,j]
            q[j,i] = q[i,j]
            
    return kernel, q
            
    

def plot_data(array, hist, bins=None, pair_list=False):
    print('Plotting array with shape --> {} and with length --> {}'.format(np.shape(array), len(array)))
    if (pair_list):
        xs = [x[0] for x in array]
        ys = [x[1] for x in array]
        sns.scatterplot(xs, ys)
        plt.show()
    else:
        if (hist is not True):
            plt.plot(array, '.')
            plt.show()
        else:
            plt.hist(array, bins=bins)
            plt.show()
            
def plot_3d_data(array1, array2):
    
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.scatter3D([x[0] for x in array1], [y[1] for y in array1], [z[2] for z in array1], label='Circles')  
    
    ax.scatter3D([x[0] for x in array2], [y[1] for y in array2], [z[2] for z in array2], label='Stars')  
    ax.legend(loc='best')
    plt.show()
    
        
    
if __name__ == "__main__":
    epochs = 50
    learning_rate = 0.0001
    l = 0.01
    circles, stars = load_data()
    np.random.seed(1)
    
    ai = np.random.random(size=len(stars)) * 0.1
    bj = np.random.random(size=len(circles)) * 0.1 
    
    circles_original = circles
    stars_original = stars
    
    for pair in circles:
        pair.append(np.sqrt((pair[0]**2) + (pair[1]**2)))
        
    for pair in stars:
        pair.append(np.sqrt((pair[0]**2) + (pair[1]**2)))
    # plot_3d_data(circles, stars)
    
    
    # kernel, q = kernel_calculation(circles_original, stars_original, 0.1)
    
    result = f_hat([stars_original[0]], stars_original, circles_original, ai, bj, 0.01)
    print(result)
    
