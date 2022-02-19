import numpy as np


class DunnIndex:
    centroids = []
    clusters = []
    k = 3

    def __init__(self, centroids, clusters, k):
        self.clusters = clusters
        self.centroids = centroids
        self.k = k

    def min_separation(self):
        # More than one cluster
        if len(self.centroids) > 1:
            min_distance = np.sqrt((self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2)
            for i in range(self.k):
                j = i + 1
                while j != self.k:
                    dist = np.sqrt((self.centroids[i][0] - self.centroids[j][0]) ** 2 + (self.centroids[i][1] - self.centroids[j][1]) ** 2)
                    print(dist)
                    if dist < min_distance:
                        min_distance = dist
                    j += 1
        # Only one cluster, the distance will be 0
        else:
            min_distance = 0

        print(min_distance)
        return min_distance

    def max_compactness(self):
        max_distance = 0
        for i in range(self.k):
            j = 0
            while j != (len(self.clusters[i]) - 1):
                dist = np.sqrt(
                    (self.clusters[i][j][0] - self.clusters[i][j + 1][0]) ** 2 + (self.clusters[i][j][1] - self.clusters[i][j + 1][1]) ** 2)
                if dist > max_distance:
                    max_distance = dist
                j += 1
        print(max_distance)
        return max_distance

    def getDunnIndex(self):
        print("Dunn Index:")
        dunn_index = self.min_separation() / self.max_compactness()
        print(dunn_index)