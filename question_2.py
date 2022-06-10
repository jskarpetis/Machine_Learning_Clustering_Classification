from mat4py import loadmat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def gaussian_kernel(X, Y, h):
    return np.exp((-1 / h) * ((X[0] - Y[0])**2 + (X[1] - Y[1])**2))
    

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
            
def plot_3d_data(array1, array2, circle_results, star_results):
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.scatter3D([x[0] for x in array1], [y[1] for y in array1], [z[2] for z in array1], label='Circles')  
    
    ax.scatter3D([x[0] for x in array2], [y[1] for y in array2], [z[2] for z in array2], label='Stars')

    ax.plot3D([(x1[0]+x2[0])/2 for x1,x2 in zip(circle_results, star_results)], [(y1[1]+y2[1])/2 for y1,y2 in zip(circle_results,star_results)], [(z1[2]+z2[2])/2 for z1,z2 in zip(circle_results,star_results)], label='Frontier',color='black')  
    ax.legend(loc='best')
    plt.show()
    
        
    
if __name__ == "__main__":
    epochs = 50
    l = 0.5
    h = 0.5
    circles, stars = load_data()

    full_set = circles + stars
    labels = [-1 for _ in circles] + [1 for _ in stars]
    ai = [ np.random.uniform(-1.0, 1.0) for _ in range(len(stars))]
    bj = [ np.random.uniform(-1.0, 1.0) for _ in range(len(circles))]   
    
    circles_original = circles
    stars_original = stars
    
    for pair in circles:
        pair.append(np.sqrt((pair[0]**2) + (pair[1]**2)))
        
    for pair in stars:
        pair.append(np.sqrt((pair[0]**2) + (pair[1]**2)))
    
    # plot_3d_data(circles, stars)
    
    for epoch in range(epochs):
        for X in full_set:
            cost = cost_function(stars_original, circles_original, X, l, h, ai, bj)
            
            for star,index in zip(stars_original,range(len(stars_original))):
                f_hat_res = f_hat(star,stars_original, circles_original, ai, bj, h)
                if f_hat_res < 0.1:
                    ai[index] += 1
                    
            for circle,index in zip(circles_original,range(len(stars_original))):
                f_hat_res = f_hat(circle,stars_original, circles_original, ai, bj, h)
                if f_hat_res > -0.1:
                    bj[index] -= 1
                    
            print('Cost --> {}'.format(cost))

    circle_results = []
    star_results = []
    
    for X, label in zip(full_set, labels):
        result = f_hat(X, stars_original, circles_original, ai, bj, h)
        circle_results.append(X + [result]) if label == -1 else star_results.append(X + [result])
        if result > 1:
            result = 1.0
        elif result < -1:
            result = -1.0
        
        if (label == -1): data = 'circle'
        else: data = 'star'
        print(f"Predicted: {result:.2f}    label: {label:.2f}")
    
    print('ai matrix -> \n{}'.format(ai))
    print('bj matrix -> \n{}'.format(bj))
    plot_3d_data(circles, stars, circle_results, star_results)
