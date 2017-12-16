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


#read in DRAMP data
def read_data_to_arr(filename):
    data = []
    with open('dramp_data_antibact_alpha.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
                data.append(row)
    
    data_arr = np.array(data)
    return data_arr
#takes in data array and subsets unique sequences
# index represents the column in the data file where sequences are stored
def subset_unique_seqs(arr, index):
    unique_seqs, inds = np.unique(arr[:, index], return_index = True)
    unique_data = arr[inds,:]
    return unique_seqs, unique_data

#returns an array of 1's and 0's where 1 denotes antifungal, and 0 denotes antibacterial
def replace_labels(data, annotation):
    labels = np.array([annotation in data[i,-1] for i in range(np.shape(data)[0])]).astype(int)
    return labels

 # for eventually combining our data. Returns the unique sequences
 # found in the file and their labels.
 # index represents the column in unprocessed data where sequences can be found
def seqs_and_labels(filename, annotation, index):
    seqs_unique, data_unique = subset_unique_seqs(read_data_to_arr(filename), index)
    #Assumes labels are in the last column of the file. Replace them with 0/1 
    #based on whether they contain antifungal activity
    labels_unique = replace_labels(data_unique, annotation)
    return seqs_unique, labels_unique
