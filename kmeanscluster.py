import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

#data = pd.read_csv('twogaussians.csv',header=None)
#data = pd.read_csv('twospirals.csv',header=None)
#data = pd.read_csv('halfkernel.csv',header=None)
#data = pd.read_csv('clusterincluster.csv',header=None)

print("classifier: K-means clustering\n")
#'twogaussians.csv','twospirals.csv','halfkernel.csv','clusterincluster.csv'
datasets = ['clusterincluster.csv']
for i in range(len(datasets)):
    data = pd.read_csv(datasets[i],header=None)
    data.name = str(datasets[i])

    data.columns = ['a','b','class']
    X = np.array(data.drop(['class'],1))
    Y = np.array(data['class'])
    X = preprocessing.scale(X)
    for c in range(len(Y)):
            if(Y[c] == 1):
                Y[c] = 0
            else : Y[c] = 1

    #0 = positive, 1=negative

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    n = 5        
    classifier = KMeans(n_clusters=n)
    classifier.fit(X)
    centroids = classifier.cluster_centers_
    labels = classifier.labels_
    #print(labels)
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = classifier.predict(predict_me)
        
        if Y[i] == prediction[0] == 0:
           TP += 1              
        if Y[i] == prediction[0] == 1:
            TN += 1
        if Y[i] == 1 and prediction[0] == 0:
            FP += 1
        if Y[i] == 0 and prediction[0] == 1:
            FN += 1
    try:
        ppv =(TP/(TP+FP))
    except ZeroDivisionError:
        ppv =(0)
    try:
        npv = (TN/(TN+FN))
    except ZeroDivisionError:
        npv = (0)
    try:
       specificity = (TN/(TN+FP))
    except ZeroDivisionError:
        specificity = 0
    try:
       sensitivity = (TP/(TP+FN))
    except ZeroDivisionError:
        sensitivity = 0
    #accuracy = ((TP+TN)/len(X))
    #accuracy = classifier.score(X)
    #score = metrics.accuracy_score(Y,classifier.predict(X))
##
##    print("dataset: ",data.name)
##    print("ppv: ",ppv)
##    print("npv: ",npv)
##    print("specificity: ",specificity)
##    print("sensitivity: ",sensitivity)
    #print("accuracy: ",score)
##    print('\n')

##    colors = ['c.','r.','c.','b.','k.']
##    markers = ['o','x','*']
##    for i in range(len(X)):
##        plt.plot(X[i][0],X[i][1],colors[Y[i]],marker= markers[labels[i]])
##        #plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize = 10)
##    plt.scatter(centroids[:,0],centroids[:,1], marker = '*', c = 'k', s=200)
##    plt.show()
##
    colors = ['g','r','c','b','k','m','yellow','orchid','fuchsia','lightcoral']
    markers = ['o','x','*','^','1','p','D','8','s','P']
    for i in range(len(X)):
        plt.plot(X[i][0],X[i][1],color=colors[labels[i]],marker= markers[labels[i]])
    plt.scatter(centroids[:,0],centroids[:,1], marker = '*', c = 'lime', s=200)
    plt.title('Number of clusters %d'%n)
    plt.show()

    #Plotting the samples
##    colors = ['w','b','r']
##    markers = ['*','x','o']
##    for i in range(len(X)):
##        plt.scatter(X[i][0],X[i][1],c = colors[Y[i]],marker= markers[Y[i]] )
##    plt.show()
        

