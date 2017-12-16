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
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn import svm
from propy import AAComposition as AAC
from propy import CTD
from propy import PseudoAAC as PAAC
from propy import QuasiSequenceOrder as QSO
from propy.PyPro import GetProDes

#perform cross validated feature selection, plot CV score vs. number of features
# returns new feature matrix with reduced number of features
def featureSelect(model, all_features, all_labels, dev_size):
    #try adding recursive subsets of feature space, 1 feature at a time
    select = fs.RFECV(SVM_linl1, step=1, cv=StratifiedShuffleSplit(n_splits=15, test_size=DEV_size, random_state=0),
                    scoring = 'accuracy')
    select.fit(all_features, all_labels)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (mean accuracy)")
    plt.title("Optimal features for linear SVM w/ l1 penalty")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    #return the new feature matrix with reduced number of optimal features
    return select.transform(all_features)

#try feature selection using boostrap aggregation: train n linear SVM's with l1 penalty,
# by randomly drawing TRAIN_SIZE samples and training model using all 1400 features with replacement
def baggedPredictors(model, all_features, all_labels, train_size, n):
    bags = BaggingClassifier(SVM_linl1, n_estimators = n, max_samples = train_size, max_features = 1400,
                        bootstrap = False, bootstrap_features = False)
    bags.fit(all_features, all_labels)
    print 'Number of unique subsamples (should be 15): ', len(np.unique(np.array(bags.estimators_samples_)))
    #store the selected features from each of the n bagged models in this list
    trunc_features = []
    #list of n SVM models which we can then extract nonzero features from
    for i, model in enumerate(bags.estimators_):
        fs_l1 = fs.SelectFromModel(model, prefit=True)
        trunc = fs_l1.transform(train_dev_features)
        #print '# of trunc features in %dth model: ' % i, len(trunc)
        trunc_features.append(trunc)
    trunc_features = np.array(trunc_features)
    #now return intersection of all truncated features
    #inter = trunc_features[0]
    #print 'shape of inter: ', inter.shape
    #for trunc in trunc_features:
    #    inter = np.intersect1d(inter, trunc)
    bagged_features = reduce(np.intersect1d, (trunc_features))
    return np.array(trunc_features), bagged_features

#build model n times using random train/dev splits each time, and extract nonzero features each round
# returns intersection of these truncated features (nonzero features that were pulled out every round)
def cv_featureSelect(model, all_features, all_labels, dev_size, n):
    #trunc_features is a n x 1400 array
    # element ij is True if the jth feature from the ith model has non-zero weight 
    trunc_features = np.ones((n, all_features.shape[1]), dtype=bool)
    for i in range(n):
        train_features, dev_features, train_labels, dev_labels = train_test_split(
                all_features, all_labels, test_size=dev_size, stratify=all_labels)
        model.fit(train_features, train_labels)
        #fs_l1 = fs.SelectFromModel(model, prefit=True)
        #trunc = fs_l1.transform(train_dev_features)
        #create logical array where True if feature j is non-zero
        trunc_inds = model.coef_[0,:]!=0
        trunc_features[i, :] = trunc_inds
    #print average number of pulled out features each round
    num_trunc = np.sum(trunc_features, axis=1)
    #print 'Average # of selected features: %.2f(+/- %0.2f)' % (np.mean(num_trunc), 2*np.std(num_trunc))
    #choose the features that are 'True' for all n models
    select_inds = np.sum(trunc_features, axis=0) == n
    selectedfeatures = all_features[:, select_inds]
    return select_inds, selectedfeatures, 
        