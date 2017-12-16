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
from sklearn.manifold import TSNE
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

#Features so far are (see Ernest's Paper and this supplementary paper, which explains the parameters:
# http://www.pnas.org/content/suppl/2016/11/09/1609893113.DCSupplemental/pnas.1609893113.sapp.pdf
# AA composition (20 features)
# sequence length (1 feature)
# net charge (1 feature)
# Solvent Accessibility, 25% of buried amino acids (1 feature) 
# Sequence Order Coupling Number for separation of 2 peptides, using Grantham distance matrix (Tau^G_2) (2 features)
# Sequence Order Coupling Number for separation of 4 peptides, using Grantham distance matrix (Tau^G_4) (4 features)



### CONSTRUCT FEATURE MATRIX ###

#print(QSO.GetSequenceOrderCouplingNumberGrant(unique_seqs[0],2))
#print(PAAC.GetAPseudoAAC(unique_seqs[0],lamda =9,weight=.05)) #yes, they misspelled 
# the pseudoAAC generalization at tier k = 9 and k= 30 parameters from the paper appear to be calculated differently
# (ie it is a some combination of propy functions and maybe original code). May want to ask Ernest about them

#create dictionary of charges for each amino acid in {A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y}
AAcharges = dict.fromkeys(['H', 'K', 'R'], 1.) #positively charged AA's
AAcharges.update(dict.fromkeys(['D', 'E'], -1.)) #negatively charged AA's
AAcharges.update(dict.fromkeys(['A','C','F','G','I','L','M','N','P','Q','S','T','V','W','Y'], 0.)) #neutral AA's
    
#compute net charge of amino acid
def getCharge(sequence):
    charges = [AAcharges[sequence[i]] for i in range(len(sequence))]
    return np.sum(charges)

def getAllPropyFeatures(sequence, minlenseq):
    descriptors = GetProDes(sequence)
    res={}
    res.update(descriptors.GetDPComp())
    res.update(descriptors.GetAAComp())
    #print(len(res.values()))
    res.update(descriptors.GetMoreauBrotoAuto())
    res.update(descriptors.GetMoranAuto())
    res.update(descriptors.GetGearyAuto())
    #print(len(res.values()))
    res.update(descriptors.GetCTD())
    #print(len(res.values()))
    res.update(descriptors.GetPAAC(lamda = minlenseq-1))
    res.update(descriptors.GetAPAAC(lamda = minlenseq-1))
    res.update(descriptors.GetSOCN(maxlag = minlenseq-1))
    res.update(descriptors.GetQSO(maxlag = minlenseq-1))
    return np.array(res.values())

def constructAllFeatures(sequences, minseqlen_ext = -1):
    minseqlen = len(min(sequences,key = len)) if minseqlen_ext == -1 else minseqlen_ext
    print 'minseqlen = ', minseqlen
    feature_matrix = []
    skipped_examples = []
    i = 0
    for seq in sequences:
        offset = [1.0]
        seqlen = [len(seq)]
        if ('X' in seq):
            seq = seq.replace('X','')
        netCharge = [getCharge(seq)]
        #descriptor_values = getAllPropyFeatures(seq, minseqlen)
        #can't use getALL() since individual descriptors and number of parameters depend on sequence len		
    	try:
    	    descriptor_values = getAllPropyFeatures(seq, minseqlen)
    	except:
    	    skipped_examples.append(i)
    	    descriptor_values = np.ones(1373)
    	features = np.concatenate((offset, seqlen, netCharge, descriptor_values))
        feature_matrix.append(features)
        if (i % 40 == 0):
             print('calculated features for %d examples' % i)
        if (i >= 2120):
             print('calculated features for %d examples' % i)
        i += 1
    #filter out features with 0 variance across examples
    selector = fs.VarianceThreshold()
    feature_subset = selector.fit_transform(feature_matrix)
    #print 'features with nonzero variance: ', feature_subset.shape
    return feature_matrix, skipped_examples

def performPCA(X, n_components):
    pca = PCA(n_components=n_components)
    newX = pca.fit_transform(X)
    varianceR = pca.explained_variance_ratio_
    return stats.zscore(newX, axis = 0, ddof  = 1), varianceR, pca.components_
    
def performtSNE(X, n_components= 3, perplexity = 30):
    tSNE = TSNE(n_components=n_components)
    return stats.zscore(tSNE.fit_transform(X), axis=0, ddof = 1)

def zscore(X):
    # z-score all descriptors so as to keep everything in the same range
    #feature_zscore = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    feature_zscore = stats.zscore(X, axis=0, ddof=1)
    return feature_zscore