#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as data
import numpy as np
from matplotlib import pyplot as plt
#from copy import deepcopy
import random
import math

def euclideanDistance(instance1, instance2, length):
    distance = 0
    #print(instance1)
    #print('prniting instance length',len(instance1),len(instance2))
    for x in range(length):
        #print(instance1[x])
            temp =(instance1[x] - instance2[x])
            distance =distance + pow(temp, 2)
       # print(len(distance))
   # print(distance)
    return math.sqrt(distance)

# Reading the data from the iris and adding the feature names

featureName = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'flowerClass']

irisData = data.read_csv('iris_data.csv',header = None,names = featureName)


X = irisData.values[:, 0:4]
Y = irisData.flowerClass.map({'Iris-setosa':1.0,'Iris-versicolor':2.0,'Iris-virginica':3.0})

# Number of clusters
k = 3
# Number of training data
r,c = X.shape
# Number of features in the data
index= random.randint(0, 49)
centroid_setosa = X[index]
index=random.randint(50, 100)
centroid_versicolor = X[index]
index=random.randint(100, 150)
centroid_virginia = X[index]

centers = np.row_stack((centroid_setosa,centroid_versicolor,centroid_virginia))

print("Initial Centroid assumed=",centers)
# Plot the data and the centers generated as random


old_centroid_setosa=centroid_setosa
old_centroid_versicolor=centroid_versicolor
old_centroid_virginia=centroid_virginia
flag=10 # The number of iteration to be done
distance=[]
cluster=[]
# When, after an update, the estimate of that center stays the same, exit loop
while flag >= 0:
    # Measure the distance to every center
    for j in range(r):
        #distance[j]=3000
        distance.append(())
        cluster.append(())
        for i in range(k):
            temp = euclideanDistance(X[j] , centers[i], 4)
            if(i==0):
                distance[j]=temp
                cluster[j]=centers[i]
            elif(temp<distance[j]):
                distance[j]=temp
                cluster[j]=centers[i]
   # print(cluster)
    updated_setosa=cluster[0:50]
    updated_versicolor=cluster[50:100]
    updated_virginia = cluster[100:150]
    new_centroid_setosa=[]
    new_centroid_versicolor=[]
    new_centroid_virginia=[]
    for x in range(4):
        a=0
        b=0
        c=0
        d=0
        new_centroid_setosa.append(())
        new_centroid_versicolor.append(())
        new_centroid_virginia.append(())
        for y in range(49):
            a+=updated_setosa[y][x]
            b+=updated_versicolor[y][x]
            c+=updated_virginia[y][x]
            #print(y)
        new_centroid_setosa[x] = a/50
        new_centroid_versicolor[x]= b/50
        new_centroid_virginia[x]= c/50
    #print(new_centroid_setosa,new_centroid_versicolor,new_centroid_virginia)
    
    if((np.array_equal(new_centroid_setosa,old_centroid_setosa) ) and np.array_equal(new_centroid_versicolor,old_centroid_versicolor) and np.array_equal(new_centroid_virginia,old_centroid_virginia)):
       # print("equal")
    #print(updated_centroid_setosa,updated_centroid_versicolor,updated_centroid_virginia)
        centers=np.row_stack((new_centroid_setosa,new_centroid_versicolor,new_centroid_virginia))
        flag=-1
    else:
        #print("not equal")
        old_centroid_setosa=new_centroid_setosa
        old_centroid_versicolor=new_centroid_versicolor
        old_centroid_virginia=new_centroid_virginia
        centers=np.row_stack((old_centroid_setosa,old_centroid_versicolor,old_centroid_virginia))
        flag=flag-1
   
print("Final Centroid Calculated=",centers)
colors=['red', 'brown', 'black']
for i in range(r):
    plt.scatter(X[i, 1], X[i,2], s=7, color = colors[int(Y[i])-1])
plt.scatter(centers[:,1], centers[:,2], marker='*', c='g', s=150)
