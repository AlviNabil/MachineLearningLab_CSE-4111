import enum
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(49)

def euclidean_distance(x1:int, x2:int):
    return np.sqrt(np.sum((x1-x2)**2))


class KMeans:
    def __init__(self, K=5, max_iter=100, plot_steps=False):
        self.K=K
        self.max_iter = max_iter
        self.plot_steps = plot_steps

        #list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        #mean feature vector of each cluster    
        self.centroids = []

    def predict(self, X: np.ndarray):
        self.X = X
        self.n_samples, self.n_features = X.shape

        #initialization of the centroids
        random_sample_idx = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idx]

        #optimizations of the centroids
        for _ in range(self.max_iter):
            #updating the clusters
            self.clusters = self._create_clusters(self.centroids)
            #updating the centroids
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()
            #check if converged
            if self._is_converged(old_centroids):
                print("Loop Count: ", _)
                break
        #return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster: 
                labels[sample_idx] = cluster_idx
        return labels
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample):
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids
    def _is_converged(self, old_centroids):
        distances = [euclidean_distance(old_centroids[i], self.centroids[i]) for i in range(self.K)]
        return sum(distances)==0
    
    def plot(self):
        figure, axis = plt.subplots(figsize=(12,12))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            axis.scatter(*point)
        for point in self.centroids:
            axis.scatter(*point, color="black", marker="x", linewidth=2)
        plt.show()