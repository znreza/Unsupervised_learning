import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import mixture
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
    clusters = [2,3,4,5,6,7,8,9,10]
    #index = np.linspace(1.0, 11.0, 11)
    index = []
    
    for n in clusters:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
               
        classifier = mixture.GaussianMixture(n_components=n)
        classifier.fit(X)
        #centroids = classifier.cluster_centers_
        nlabels = []
        labels = []
        #print(labels)
        for i in range(len(X)):
            predict_me = np.array(X[i].astype(float))
            predict_me = predict_me.reshape(-1, len(predict_me))
            prediction = classifier.predict(predict_me)
            nlabels.append(prediction)
            
            if Y[i] == prediction[0] == 0:
               TP += 1              
            if Y[i] == prediction[0] == 1:
                TN += 1
            if Y[i] == 1 and prediction[0] == 0:
                FP += 1
            if Y[i] == 0 and prediction[0] == 1:
                FN += 1
        for a in range(len(X)):   
            labels.append(nlabels[a][0])
        index.append(metrics.calinski_harabaz_score(X, labels))
    

##        colors = ['g','r','c','b','k','m','yellow','orchid','fuchsia','lightcoral']
##        markers = ['o','x','*','^','1','p','D','8','s','P']
##        for i in range(len(X)):
##            plt.plot(X[i][0],X[i][1],color=colors[labels[i]],marker= markers[labels[i]])
##        plt.scatter(centroids[:,0],centroids[:,1], marker = '*', c = 'lime', s=200)
##        plt.title('Number of clusters %d'%n)
##        plt.show()
##        
    print("validity of index: ", index)
    print('\n')
    #Plotting the samples
##    colors = ['w','b','r']
##    markers = ['*','x','o']
##    for i in range(len(X)):
##        plt.scatter(X[i][0],X[i][1],c = colors[Y[i]],marker= markers[Y[i]] )
##    plt.show()
        

