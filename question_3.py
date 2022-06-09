
from operator import indexOf, is_
from mat4py import loadmat
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import trange



def load_data():
    data = loadmat('./Data/data33.mat')
    x_data = data['X']
    return x_data, data


def manage_data(data):
    data = np.array(data)
    new_data = np.reshape(data, (200,2))
    return new_data


def plot_data(arrays, pair_list=False, centroids=False, centroid_list=None):
    if (pair_list):
        for matrix in arrays:
            xs = [x[0] for x in matrix]
            ys = [x[1] for x in matrix]
            sns.scatterplot(xs, ys, label=f"{indexOf(arrays, matrix)} cluster")
            if (centroids):
                xs = [x[0] for x in centroid_list]
                ys = [x[1] for x in centroid_list]
                sns.scatterplot(xs, ys, s=80, color='Black')
        plt.legend(loc='best')
        plt.show()

            
def plot_3d_data(data):
    
    local_data = data.tolist()
    for pair in local_data:
        pair.append(np.sqrt((pair[0]**2) + (pair[1]**2)))
        
    array1 = local_data[:100]
    array2 = local_data[100:]
    
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.scatter3D([x[0] for x in array1], [y[1] for y in array1], [z[2] for z in array1], label='first-100-group-1')  
    
    ax.scatter3D([x[0] for x in array2], [y[1] for y in array2], [z[2] for z in array2], label='second-100-group-2')  
    ax.legend(loc='best')
    plt.show()
            
            
def plot_3d_clusters(clusters, centroids):
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    for cluster_id in clusters:
        cluster_data = clusters[cluster_id]
        ax.scatter3D([x[0] for x in cluster_data], [y[1] for y in cluster_data], [z[2] for z in cluster_data], label=f'{cluster_id} cluster')  
    for centroid_id in centroids:
        centroid = centroids[centroid_id]
        ax.scatter3D([centroid[0]], [centroid[1]], [centroid[2]], s = 100, label=f'{centroid_id} centroid') 
    ax.legend(loc='best')
    plt.show()
        
            
def k_means(n_clusters, dataset, epochs, is_3d=False):
    n = len(dataset)
    centroids = {}
    clusters = {}
    variance_array = []
    
    
    # Initial clusters
    for cluster in range(n_clusters):
        clusters[cluster] = []
        if (is_3d is False):
            centroids[cluster] = [np.random.randn(), np.random.randn()]
        else:
            centroids[cluster] = [np.random.randn(), np.random.randn(), np.random.randn()]
    
    
    ### Algorithm to minimize the total_variance###
    for epoch in trange(epochs):
        total_variance = 0
        for point in dataset:
            # print("\nData point --> {}\n".format(point))
            stored_norms = []
            
            for cluster_id in centroids:
                centroid_coord = centroids[cluster_id]
                norm_squared = np.linalg.norm(np.subtract(point,centroid_coord))**2
                # print('Norm squared --> {}'.format(norm_squared))
                stored_norms.append(norm_squared)
            
            minimum_norm_index = np.argmin(stored_norms)
            # print(minimum_norm_index)
            clusters[minimum_norm_index].append(point.tolist())
            
            total_variance += stored_norms[minimum_norm_index]
            # print(total_variance)
            
        for cluster_id in clusters:
            cluster_data = clusters[cluster_id]
            
            if (is_3d is False):
                best_centroid = np.zeros(shape=(2,))
            else:
                best_centroid = np.zeros(shape=(3,))
                
            for point in cluster_data:
                best_centroid = np.add(best_centroid, point)
            best_centroid = np.divide(best_centroid, len(cluster_data))
            centroids[cluster_id] = best_centroid.tolist()      
        
        variance_array.append(total_variance)  
        
    return variance_array, clusters, centroids
        
        


def calculate_cost(predicted_clusters):
        
    
    error = 0
    correct_matrix = np.zeros(shape=(len(predicted_clusters),))
    for i in range(100):
        correct_matrix[i] = 1
    
    subtracted_list = np.subtract(correct_matrix, predicted_clusters)
    
    for element in subtracted_list:
        if element != 0:
            error += 1
    
    error_percentage = error / len(predicted_clusters) * 100
    
    print('Total mistakes --> {}\n'.format(error))
    print('Error --> {} %\n'.format(error_percentage))
    
    return error_percentage

if __name__ == "__main__":
    np.random.seed(1)
    x_data, all_data = load_data()
    

    # print(x_data, '\n')
    # print(all_data.keys(), '\n')
    # print(all_data.values(), '\n')
    # print(x_data.shape)
    
    data = manage_data(x_data)
    
    
    #####################################TRIALS###################################################
    ##############################################################################################

    # print(data, '\n')
    # print(data.shape)
    
    # PLotting all data together
    # plot_data([data], pair_list=True)

    plot_3d_data(data)
    
    #######################################################2-D KMEANS################################################################
    
    total_variance, clusters, centroids = k_means(n_clusters=2, dataset=data, epochs=50)
    
    print(total_variance, '\n')
    
    clusters_as_arrays = []
    for key in clusters:
        clusters_as_arrays.append(clusters[key])
    
    centroids_as_arrays = []
    for key in centroids:
        centroids_as_arrays.append(centroids[key])
    
    plot_data(clusters_as_arrays, pair_list=True, centroids=True, centroid_list=centroids_as_arrays)
    
    #######################################################3-D KMEANS################################################################
    
    data = data.tolist()
    for pair in data:
        pair.append(np.sqrt((pair[0]**2) + (pair[1]**2)))
        
    total_variance, clusters, centroids = k_means(n_clusters=2, dataset=np.array(data), epochs=20, is_3d=True)
    plot_3d_clusters(clusters, centroids)
    
    # calculate_cost(clusters)
    