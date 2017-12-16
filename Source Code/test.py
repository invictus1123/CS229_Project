from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def score(models, modelnames, td_features, td_labels, test_features, test_labels):
    scores = []
    for i, model in enumerate(models):
        #use the full train dev set to train models
        model.fit(td_features, td_labels)
        yhat = model.predict(test_features)
        print(yhat)
        acc = accuracy_score(yhat,test_labels)
        print('Test accuracy for model %s is: %f' %(modelnames[i],acc))
        try:
            roc = roc_auc_score(yhat, test_labels)
        except: 
            roc = 0
        f1 = f1_score(yhat,test_labels, average = 'binary')
        avep = average_precision_score(yhat,test_labels)
        precision = precision_score(yhat,test_labels)
        recall = recall_score(yhat,test_labels)
        scores_i = { 'accuracy': acc, 'precision': precision, 'recall' : recall,  'roc_auc_score' : roc, 'average_precision_score' : avep, 'f1' : f1} 
        scores.append(scores_i)
    return scores    
    
    
