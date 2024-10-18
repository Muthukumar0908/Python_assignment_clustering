import numpy as np

# Matrix class with required methods
class matrix:
    def __init__(self, file_path):
        self.array_2d = None
        self.load_from_csv(file_path)

    def load_from_csv(self, file_path):   # conver to matrix in my csv file
          
        self.array_2d = np.loadtxt(file_path, delimiter=',')
        print("sample arrar\y_2d in file vector values:",self.array_2d[0])

    def standardise(self):   # standrdise the all data for matrix values
        mean = np.mean(self.array_2d, axis=0)
        print("mean:",mean)
        max_val = np.max(self.array_2d, axis=0)
        print("max_values:",max_val)
        min_val = np.min(self.array_2d, axis=0)
        print("min_value:",min_val)
        self.array_2d = (self.array_2d - mean) / (max_val - min_val)

    def get_distance(self, other_matrix, row_i):       # row number == row_i (row_i is input row number (vector))and matrix == other_matrix    
        #Euclidean distance in specific row
        euclidean_distance = np.sqrt(np.sum((self.array_2d[row_i] - other_matrix.array_2d) ** 2, axis=1)) # axis 1 values
        return euclidean_distance

    def get_weighted_distance(self, other_matrix, weights, row_i):     # row_i is my input row 
        #get Weight in Euclidean distance from specific row
        diff = self.array_2d[row_i] - other_matrix.array_2d
        weight_Euclidean_distance= np.sqrt(np.sum(weights * (diff ** 2), axis=1))
        return weight_Euclidean_distance

    def get_count_frequency(self):
        #Return the frequency of each element in the matrix
        unique, counts = np.unique(self.array_2d, return_counts=True)
        count_frequency=dict(zip(unique, counts))
        return count_frequency


def get_initial_weights(m):
    #Generate initial random weights
    weights = np.random.rand(m)
    initial_weight=weights / np.sum(weights)
    return initial_weight



def get_centroids(data, S, K):
    #Compute centroids for clusters
    centroids = np.zeros((K, data.array_2d.shape[1]))
    for k in range(K):
        rows_in_cluster = data.array_2d[S == k]
        if len(rows_in_cluster) > 0:
            centroids[k] = np.mean(rows_in_cluster, axis=0)
    return centroids


def get_separation_within(data, centroids, S, K):
    #Calculate separation within clusters
    a = np.zeros(data.array_2d.shape[1])
    for j in range(data.array_2d.shape[1]):
        for k in range(K):
            rows_in_cluster = data.array_2d[S == k]
            a[j] += np.sum(np.linalg.norm(rows_in_cluster[:, j] - centroids[k, j]))
    return a



def get_separation_between(data, centroids, S, K):
#Calculate separation between clusters
    b = np.zeros(data.array_2d.shape[1])
    for j in range(data.array_2d.shape[1]):
        for k in range(K):
            count_k = np.sum(S == k)
            b[j] += count_k * np.linalg.norm(centroids[k, j] - np.mean(data.array_2d[:, j]))
    return b


def get_groups(data, K):
    # Ensure that K is not greater than the number of data points
    num_rows, num_cols = data.array_2d.shape
    
    if K > num_rows:
        raise ValueError(f"Number of clusters K={K} cannot be greater than the number of data points {num_rows}.")
    
    # Initialize group assignments (S) and centroids
    S = np.zeros(num_rows, dtype=int)
    centroids = data.array_2d[np.random.choice(num_rows, K, replace=False)]
    
    print(f"Initial centroids shape: {centroids.shape}")
    
    while True:
        new_S = np.zeros(num_rows, dtype=int)  
        for i in range(num_rows):
            distances = np.linalg.norm(data.array_2d[i] - centroids, axis=1)
            # print(f"Row {i} distances to centroids: {distances}")
            new_S[i] = np.argmin(distances) 
            # print(f"Assigned cluster for row {i}: {new_S[i]}")
        
        if np.all(S == new_S):
            break
        
        S = new_S  
        # print(f"Updated cluster assignments: {S}")
        
        for k in range(K):
            cluster_points = data.array_2d[S == k]
            # print(f"Points in cluster {k}: {cluster_points}")
            
            if len(cluster_points) > 0:  # Avoid empty clusters
                centroids[k] = np.mean(cluster_points, axis=0)
                # print(f"Updated centroid {k}: {centroids[k]}")
            else:
                # print(f"Cluster {k} is empty. Reassigning to random point.")
                centroids[k] = data.array_2d[np.random.choice(num_rows)]
    
    return S

# def get_groups(data, K):
#     #Assign groups based on the nearest centroids
#     S = np.zeros(data.array_2d.shape[0], dtype=int)
#     centroids = data.array_2d[np.random.choice(data.array_2d.shape[0], K, replace=False)]
#     while True:
#         new_list=[]
#         for c in centroids:
#             norm_=np.argmin(np.linalg.norm(data.array_2d - c, axis=1))
#             new_list.append(norm_)
#         print(new_list)
#         new_S = np.array(new_list)
#         if np.all(S == new_S):
#             break
#         S = new_S
#     return S


def get_new_weights(data, centroids, weights, S, K):
    #Update the weights vector
    a = get_separation_within(data, centroids, S, K)
    b = get_separation_between(data, centroids, S, K)
    new_weights = 0.5 * (weights + (b / a) / np.sum(b / a))
    return new_weights


# Test function with file path
def run_test(file_path):
    #Run tests in custom file path
    m = matrix(file_path)
    m.standardise()
    for k in range(2, 11):
        for i in range(20):
            S = get_groups(m, k)
            print(f'K={k}, Frequency: {m.get_count_frequency()}')


file_path = "C:\\Users\\murugan\\Pictures\\anubavam\Data (2).csv"

run_test(file_path)