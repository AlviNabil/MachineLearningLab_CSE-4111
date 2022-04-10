import numpy as np
import pandas as pd

def euclideanDistance(instance1, instance2, featureDimension):
    distance = 0
    for i in range(featureDimension):
        distance += pow(instance1[i]-instance2[i],2)
    return np.sqrt(distance)

class simpleKNN:
    def __init__(self, k_neighbors: int):
        self.k = k_neighbors
    

    def fit(self, trainFeatures, trainTargets) -> None:
        self.trainSet =  pd.merge(trainFeatures, trainTargets, left_index=True, right_index=True, how="left")
       
    def getNeighbors(self, testInstance):
        k = self.k
        distances = []
        trainSet = self.trainSet
        length = trainSet.shape[1]-1
        for i in range(trainSet.shape[0]):
            dist = euclideanDistance(testInstance, trainSet.iloc[i].values, length)
            # print(dist)
            distances.append((trainSet.iloc[i].values, dist))
        distances = pd.DataFrame(distances)
        distances=distances.sort_values(distances.columns[1])
        neighbors = []
        for i in range(k):
            neighbors.append(distances.iloc[i])
            # print(1)
        return neighbors

    def getPredictionValue(self, testInstance):
        ans = []
        neighbors = self.getNeighbors(testInstance)
        neighbors = pd.DataFrame(neighbors)
        k = self.k
        for i in range(k):
            ans.append(neighbors.iloc[i].values[0][neighbors.iloc[0][0].shape[0]-1])
            # print(2)
        return max(set(ans), key=ans.count)
    
    def predict(self, testFeatures):
        predictedValues = []
        for i in range(testFeatures.shape[0]):
            predictedValues.append(self.getPredictionValue(testFeatures.iloc[i].values))

        return predictedValues