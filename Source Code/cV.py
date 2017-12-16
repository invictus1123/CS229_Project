from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import csv
import logreg
import camp
import apd
#import tensorflow as tf #for when we build NN
from functools import reduce
from scipy import stats
from sklearn import feature_selection as fs
from sklearn import linear_model
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn import svm
from propy import AAComposition as AAC
from propy import CTD
from propy import PseudoAAC as PAAC
from propy import QuasiSequenceOrder as QSO
from propy.PyPro import GetProDes
import features
import split
import feature_sel 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#implement stratified shuffled cross validation using scikit-learn
# train_dev_data contains sequences in first column and labels in second column
# dev_size is an integer representing the number of examples in the dev set OR a double specifying
#   fraction of train_dev set that should be held out during each round of CV
# n_shuffles is the number of iterations, models is a list of estimators to test, and
# model_names is a list of strings containing the names of the estimators
def crossValidate(train_dev_features, train_dev_labels, dev_size, n_shuffles, models, model_names,nl1 = -1, pca = -1, ncomponents = -1):
    #stratified shuffle split will keep unbalanced classes for the dev set but retain balance for the train set
    #aim for a 20/31 split in the dev set.
    _,_,_,_,sss = split.balTrainUnbalDevSplit(train_dev_features, train_dev_labels, n_shuffles)
    #sss = StratifiedShuffleSplit(n_splits=n_shuffles, test_size=dev_size)
    SVM_linl1_fs = svm.LinearSVC(C = 1, penalty ='l1', loss = 'squared_hinge', dual = False, random_state=0)
    model_scores = []
    model_accuracies_stds = np.zeros((len(models),4))
    td_features_cpy = np.copy(train_dev_features)
    td_labels_cpy  = np.copy(train_dev_labels)
    for i, model in enumerate(models):
        #manual cv start
        # train_acc = np.zeros(n_shuffles)
        # dev_acc = np.zeros(n_shuffles)
        # precision = np.zeros(n_shuffles)
        # recall = np.zeros(n_shuffles)
        # f1 = np.zeros(n_shuffles)
        # roc_auc = np.zeros(n_shuffles)
        # avg_p = np.zeros(n_shuffles)
        # for n in range(n_shuffles):
        #     train_features, dev_features, train_labels, dev_labels = train_test_split(
        #         train_dev_features,train_dev_labels, test_size=dev_size, stratify=train_dev_labels)
        #     if(model_names[i][-1] == 'm'):
        #         inds,_= feature_sel.cv_featureSelect(SVM_linl1_fs, train_features, train_labels, dev_size, 1)
        #         train_features = train_features[:,inds]
        #         dev_features = dev_features[:,inds]
        #     elif(nl1 !=-1):
        #         inds,_= feature_sel.cv_featureSelect(SVM_linl1_fs, train_features, train_labels, dev_size, nl1)
        #         train_features = train_features[:,inds]
        #         dev_features = dev_features[:,inds]
        #     elif(pca !=-1 and ncomponents!=-1):#PCA 
        #         train_features, _, components = features.performPCA(train_features, ncomponents)
        #         dev_features = np.dot (dev_features,components.T)
        #     model.fit(train_features, train_labels)
        #     yhat = model.predict(dev_features)
            
        #     train_acc[n] = accuracy_score(model.predict(train_features),train_labels)
        #     dev_acc[n] = accuracy_score(yhat,dev_labels)
        #     f1[n] = f1_score(yhat, dev_labels)
        #     roc_auc[n] = roc_auc_score(yhat,dev_labels)
            
        #     precision[n] =precision_score(yhat,dev_labels)
        #     recall[n] = recall_score(yhat,dev_labels)
        #     avg_p[n]=average_precision_score(yhat,dev_labels)
        # scores = {"train_accuracy" : train_acc, "test_accuracy" : dev_acc, "f1" : f1, "roc_auc" : roc_auc,
        #             "precision" : precision, "recall" : recall, "average_precision" : avg_p}
        #scores is a dictionary with keys: 'score_time', 'test_score', 'train_score', 'fit_time'
        #note that cross_validate() has a scoring parameter where you can specify more evaluation metrics
        # other than just the score (mean accuracy) computed by the estimator
        ##manual cv end
        #auto cv start
        scoringParams= { 'accuracy', 'precision','recall','f1', 'roc_auc', 'average_precision'}
        scores = cross_validate(estimator=models[i], X=train_dev_features, y=train_dev_labels, 
                           scoring = scoringParams, cv = sss, return_train_score=True)#, random_state = 0)
        #print(scores.keys())
        #auto cv end
        train_accm = scores['train_accuracy'].mean()
        train_acc_std =scores['train_accuracy'].std()
        dev_accm = scores['test_accuracy'].mean()
        dev_acc_std = scores['test_accuracy'].std()
        #print mean train and dev accuracies from all n shuffles
        print("CV %s train accuracy: %0.2f (+/- %0.2f)" % 
                 (model_names[i], scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2))
        print("CV %s dev accuracy: %0.2f (+/- %0.2f)" % 
                 (model_names[i], scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
        model_accuracies_stds[i,:] = np.array([train_accm,train_acc_std, dev_accm,dev_acc_std])
        model_scores.append(scores)
    return model_scores, model_accuracies_stds

