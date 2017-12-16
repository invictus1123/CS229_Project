from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import csv
import logreg
import camp
import string
import apd
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

import features #decomposed code
import feature_sel
import dramp
import metrics
import gS
import cV

#plotting parameters
params = {'backend': 'pdf','axes.labelsize': 12,'text.fontsize': 12,'legend.fontsize': 10,'xtick.labelsize': 10,
    'ytick.labelsize': 10,'text.usetex': False,'figure.figsize': [6,4], 'font.family' : 'sans-serif', 'font.sans-serif' : ['Arial'], 
    'mathtext.fontset': 'stixsans', 'xtick.major.pad': 2, 'ytick.major.pad': 2}



### MAIN METHOD: Leave all functions above because it's getting confusing to read the code with "main method"
### commands interspersed throughout script ###

#main methods will exit and not make -i work
#def main():

#read data
# CAMP_seqs, CAMP_labels = camp.read_CAMP()
# APD_seqs, APD_labels = apd.read_APD('APD_antifungal_165.txt', 'APD_antibacterial_393.txt')
# APD_seqs2, APD_labels2 = apd.read_APD('APD_antifungal_1047.csv', 'APD_antibacterial_2448.csv')
# DRAMP_seqs, DRAMP_labels = dramp.seqs_and_labels('dramp_data_antibact_alpha.csv', 'fungal', 3)
# #total of 655 sequences
# all_seqs = np.concatenate((CAMP_seqs, DRAMP_seqs, APD_seqs, APD_seqs2))
# all_labels = np.concatenate((CAMP_labels, DRAMP_labels, APD_labels, APD_labels2))
# #remove redundancies across data sets for a total of 2654 unique sequences
# #unique_seqs, inds = np.unique(all_seqs[string.find(all_seqs,'X') == -1], return_index=True)
# unique_seqs, inds = np.unique(all_seqs, return_index=True)


# processed_seqs = unique_seqs
# unique_labels = all_labels[inds]
# unique_labels[unique_labels == -1.] = 0. ## change -1s to 0
# num_pos = np.sum(unique_labels) #Number of positives (antifungal activity) = 1072
# #going to only use 1048 of these because code is breaking for >2120 examples
# num_per_class = 1048

# #split into positives and negatives; first column contains sequences, second column contains labels
# pos_seqs = processed_seqs[unique_labels==1]
# pos_labels = unique_labels[unique_labels==1]
# pos_examples = np.array([pos_seqs, pos_labels]).T
# neg_seqs = processed_seqs[unique_labels==0]
# neg_labels = unique_labels[unique_labels==0]
# neg_examples = np.array([neg_seqs, neg_labels]).T 
# #randomly subsample 1072 of 1582 antibacterial AMPs to create a balanced mini data set of 2144 sequences
# np.random.seed(100)
# np.random.shuffle(neg_examples) 
# neg_subset = neg_examples[0:num_pos, :]
# np.random.shuffle(pos_examples) 
# pos_subset = pos_examples[0:num_pos, :]
# #combine positive and negative examples for mini data set of 2144 examples
# mini_set = np.concatenate((pos_subset, neg_subset), axis=0)
# mini_features, skipped_inds = features.constructAllFeatures(mini_set[:,0]) #only 2140 got skipped
# mini_features = np.array(mini_features)
# #delete sequence 2140 (class 0) and sequence 0 (class 1) to keep classes balanced
# #now mini_set has 2142 sequences in total
# skipped_inds.append(0)
# mini_features_t = np.delete(mini_features, skipped_inds, axis=0) 
# mini_labels = mini_set[:,1].astype(float)
# mini_labels_t = np.delete(mini_labels, skipped_inds, axis=0) 

#seq_lens = np.array([len(s) for s in mini_set[:,0]]) #shortest peptide in entire database = 2
#compute feautres and labels of entire mini_set
#large_features = np.save('large_features', mini_features_t)
#large_labels = np.save('large_labels', mini_labels_t)

mini_features = np.load('large_features.npy') #(2142, 1376)
mini_labels = np.load('large_labels.npy') #(2142,)

#mini_features_PCA = features.performPCA(mini_features,n_components=50)
#mini_features_tSNE = features.performtSNE(mini_features,n_components=3, perplexity=10)


#split mini_set into train/dev set and test set according to the above sizes
TEST_SIZE = 214 #~10% of mini_set
DEV_SIZE = 214 #~10% of mini_set
TRAIN_SIZE = 1714 #~80% of mini_set

# stratify parameter maintains balanced classes according to labels
train_dev_features, test_features, train_dev_labels, test_labels = train_test_split(
                mini_features, mini_labels, test_size=TEST_SIZE, stratify=mini_labels, random_state = 0)
# PCA_td_features, PCA_test_features, PCA_td_dev_labels, PCA_test_labels = train_test_split(
#                 mini_features_PCA, mini_labels, test_size=TEST_SIZE, stratify=mini_labels, random_state = 0)


print 'train_dev_features size: ', train_dev_features.shape
print 'train_dev_labels size: ', train_dev_labels.shape
print 'test_features size: ', test_features.shape
print 'test_labels size: ', test_labels.shape


### MANUAL MODEL TESTING ###

### FEATURE SELECTION ###

#attempt to build SVM with l1 penalty
SVM_linl1 = svm.LinearSVC(penalty='l1', loss = 'squared_hinge', dual = False, random_state = 0)

#try cross validated feature selection by recursively searching subsets of feature space
#transformed_features = featureSelect(SVM_linl1, train_dev_features, train_dev_labels, DEV_SIZE)
#print 'Transformed features from RFECV: ', transformed_features.shape

#try finding bagged features from computing 15 SVM linl1 models
#trunc_features, bagged_features = baggedPredictors(SVM_linl1, train_dev_features, train_dev_labels, TRAIN_SIZE, n=15)
#print 'Shape of trunc_features for 15 models: ', trunc_features.shape
#print 'Shape of final bagged_features: ', bagged_features.shape

#try extracting features from SVM_linl1 just once (rather than via CV)
# threshold is 1e-5 (so features with weights less than this get truncated)
#feature_selector_l1 = fs.SelectFromModel(SVM_linl1, prefit=False)
#truncated_features = feature_selector_l1.fit_transform(train_dev_features, train_dev_labels)
#print 'Features selected from l1 SVM: ', truncated_features.shape

#extract features manually without scikit for comparison
#SVM_linl1.fit(train_dev_features,train_dev_labels )
#coeff_linl1 = SVM_linl1.coef_
#manual_features = train_dev_features[:,coeff_linl1[0,:]!=0]
#print 'Features manually selected from l1 SVM: ', manual_features.shape

### CROSS VALIDATED MODEL TESTING ###
SVM_linl1_165 = svm.LinearSVC(penalty ='l1', loss = 'squared_hinge', dual = False, random_state=0)
SVM_linl1_165.fit(train_dev_features,train_dev_labels)
coeff_linl1 = SVM_linl1_165.coef_
#884 features selected from one run
linl1_onerun_features = train_dev_features[:,coeff_linl1[0,:]!=0]
print 'Features surviving one run from l1 SVM: ', linl1_onerun_features.shape

#run cross validation on the features resulting from a single run of l1 feature selection
SVM_rbf_m = svm.SVC(kernel = 'rbf')
SVM_linl1_m = svm.LinearSVC(penalty='l1',loss = 'squared_hinge',dual= False)
SVM_linl2_m = svm.LinearSVC(penalty='l2',loss = 'squared_hinge',dual = True)
SVM_poly_m = svm.SVC(kernel = 'poly', degree = 5)
logisticl2_m = linear_model.LogisticRegression() #logreg.LogReg(its = 20000, alpha = .001, Lambda = .25)

#Run cross validation over all features
#SVM_linl1 above
SVM_rbf = svm.SVC(kernel = 'rbf')
SVM_poly = svm.SVC(kernel='poly', degree = 5)
#SVM_linl1 = svm.LinearSVC(penalty='l1', loss = 'squared_hinge', dual = True)
SVM_linl2 = svm.LinearSVC(penalty='l2', loss = 'squared_hinge', dual = True)
logisticl2 = linear_model.LogisticRegression()#logreg.LogReg(its = 5000, alpha = .001, Lambda = .25)
#abandon our crappy logistic regression code for builtin (speed up Grid Search)

#Run cross validation on the features resulting from our optimal set of features
#Bagged 299 features
inds, bagged_features = feature_sel.cv_featureSelect(SVM_linl1, train_dev_features, train_dev_labels, DEV_SIZE, n=14)
print 'Selected features: ', bagged_features.shape
SVM_rbf_s  = svm.SVC(kernel = 'rbf')
SVM_poly_s = svm.SVC(kernel = 'poly',degree = 5)
SVM_linl1_s =svm.LinearSVC(penalty = 'l1',loss = 'squared_hinge', dual = False)
SVM_linl2_s = svm.LinearSVC(penalty='l2',loss= 'squared_hinge', dual = True)
logisticl2_s = linear_model.LogisticRegression()#logreg.LogReg(its = 40000, alpha = .001, Lambda = .25)

# SVM_rbf_PCA  = svm.SVC(kernel = 'rbf')
# SVM_poly_PCA = svm.SVC(kernel = 'poly',degree = 5)
# SVM_linl1_PCA =svm.LinearSVC(penalty = 'l1',loss = 'squared_hinge', dual = False)
# SVM_linl2_PCA = svm.LinearSVC(penalty='l2',loss= 'squared_hinge', dual = True)
# logisticl2_PCA = linear_model.LogisticRegression()

classifiers = [SVM_rbf, SVM_poly, SVM_linl1, SVM_linl2, logisticl2]
classifiers_m = [SVM_rbf_m, SVM_poly_m, SVM_linl1_m, SVM_linl2_m, logisticl2_m]
classifiers_s = [SVM_rbf_s, SVM_poly_s, SVM_linl1_s, SVM_linl2_s, logisticl2_s]
#classifiers_PCA = [SVM_rbf_PCA, SVM_poly_PCA, SVM_linl1_PCA, SVM_linl2_PCA, logisticl2_PCA]



names = ['SVM_rbf', 'SVM_poly','SVM_lin1', 'SVM_linl2', 'logisticl2']
names_m = ['SVM_rbf_m', 'SVM_poly_m','SVM_lin1_m', 'SVM_linl2_m', 'logisticl2_m']
names_s = ['SVM_rbf_s', 'SVM_poly_s','SVM_lin1_s', 'SVM_linl2_s', 'logisticl2_s']
#names_PCA = ['SVM_rbf_PCA', 'SVM_poly_PCA','SVM_lin1_PCA', 'SVM_linl2_PCA', 'logisticl2_PCA']

cv_scores, accstds = cV.crossValidate(train_dev_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
                                models=classifiers, model_names=names)
# cv_scores_m, accstds_m = cV.crossValidate(linl1_onerun_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
#                                 models=classifiers_m, model_names=names_m)
# cv_scores_s, accstd_s = cV.crossValidate(bagged_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
#                                 models=classifiers_s, model_names=names_s)
#cv_scores_p, accstd_p = cV.crossValidate(PCA_td_features, PCA_td_dev_labels, DEV_SIZE, n_shuffles = 15, \
#                                 models=classifiers_PCA, model_names=names_PCA)

##plot tSNE features against each other to observe distribution
# plt.figure(1)
# plt.clf()
# plt.plot(mini_features_tSNE[mini_labels==1,0], mini_features_tSNE[mini_labels==1,2],'rx',label='positive')
# plt.plot(mini_features_tSNE[mini_labels!=1,0], mini_features_tSNE[mini_labels!=1,2],'bo', label='neg')
# plt.legend()
# plt.title('tSNE feature 0 vs. 2')
# plt.xlabel('feature 0')
# plt.ylabel('feature 2')
# plt.savefig('tSNE0v2.png')

# plt.figure(2)
# plt.clf()
# plt.plot(mini_features_tSNE[mini_labels==1,1], mini_features_tSNE[mini_labels==1,2],'rx',label='positive')
# plt.plot(mini_features_tSNE[mini_labels!=1,1], mini_features_tSNE[mini_labels!=1,2],'bo', label='neg')
# plt.title('tSNE feature 1 vs. 2')
# plt.xlabel('feature 1')
# plt.ylabel('feature 2')
# plt.legend()
# plt.savefig('tSNE1v2.png')



### Experiment for finding the best number of L1 classifiers to use for feature selection                                
#maximum number of splits to try bagging
# N = 20
# nfeatures = np.zeros(N)
# accuracies_stds_over_N = np.zeros((len(names_subsetted),4,N))#4 for train and test accuracy and std
#3d array: row = model, col = metric (tr_acc, tr_std, dev_acc, dev_std), 
#           page = number of shuffles used to come up with features
# for it in range(1,N + 1):
#     inds, bagged_features = feature_sel.cv_featureSelect(SVM_linl1, train_dev_features, train_dev_labels, DEV_SIZE, n=it)
#     print 'Selected features for n = %d: ' % it, bagged_features.shape
#     cv_scores_subsetted,accstds = crossValidate(bagged_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
#                                 models=classifiers_subsetted, model_names=names_subsetted)
#     accuracies_stds_over_N[:,:,it-1] = accstds
#     nfeatures[it-1] = bagged_features.shape[1]
    
#Plot each model's progress over the number of (folds of feature selection, or number of features?)
# for model in range(len(names_subsetted)):
#     train_acc_over_N = accuracies_stds_over_N[model,0,:]
#     train_std_over_N = accuracies_stds_over_N[model,1,:]
#     dev_acc_over_N = accuracies_stds_over_N[model,2,:]
#     dev_std_over_N = accuracies_stds_over_N[model,3,:]
#     plt.figure(200 + 2*model)
#     plt.errorbar(range(1,N+ 1),train_acc_over_N, xerr= None, yerr = train_std_over_N, label = 'train')
#     plt.errorbar(range(1,N+ 1),dev_acc_over_N, xerr= None, yerr = dev_std_over_N, label = 'dev')
#     plt.title('Mean CV (k = 15) Train/Dev Set Accuracy for %s vs. N_rounds' % names[model])
#     plt.xlabel('Number of Classifiers Before Feature Selection')
#     plt.legend(loc = 'lower right')
#     plt.ylabel('Accuracy')
#     plt.savefig('%s.png' % names[model])
#     plt.figure(200 + 2*model + 1)
#     plt.errorbar(nfeatures,train_acc_over_N,xerr=None,yerr=train_std_over_N,label='train')
#     plt.errorbar(nfeatures,dev_acc_over_N,xerr=None,yerr=dev_std_over_N,label='dev')
#     plt.title('Mean CV (k = 15) Train/Dev Set Accuracy for %s vs. N_features' % names[model])
#     plt.xlabel('Number of Features')
#     plt.legend(loc = 'lower right')
#     plt.ylabel('Accuracy')
#     plt.savefig('%s_vsfeatures.png' % names[model])
    
    
# MM = metrics.constructMetricsTable(names,cv_scores)
# MM_m = metrics.constructMetricsTable(names_m,cv_scores_m)
# MM_s = metrics.constructMetricsTable(names_s,cv_scores_s)
# np.savetxt("Metrics_AllFeatures.csv", MM,fmt = '%s', delimiter=",")
# np.savetxt("Metrics_SingleL1.csv", MM_m,fmt = '%s', delimiter=",")
# np.savetxt("Metrics_NOpt.csv", MM_s, fmt = '%s',delimiter=",")

#gS.gridSearch(train_dev_features, train_dev_labels, DEV_SIZE)
#gS.gridSearch(linl1_onerun_features, train_dev_labels, DEV_SIZE)
#gS.gridSearch(bagged_features, train_dev_labels, DEV_SIZE)



### RESULTS from CV experiments ###

# Experiment 1: CV on SVM w/ l1 penalty, dev_size = 42 (80-10-10 train-dev-test split), k=5
#     Results:    CV dev set accuracy scores:  [ 0.76190476  0.83333333  0.5952381   0.80952381  0.76190476]
# Experiment 2: CV on SVM w/ l1 penalty, dev_size = 32 (80-20 train-dev split), k=15
#     Results:    CV train set mean accuracy: 0.98 (+/- 0.01)
#                 CV dev set mean accuracy: 0.70 (+/- 0.12)
# Experiment 3: Train SVM w/ both l1 and l2 penalties; dev_size = 32, k= 15, eliminated random shuffle of
#     negative examples so that mini set of 406 examples is constant. only randomness then is the train-dev-test split
#     Results:    CV SVM_linl1 train accuracy: 0.98 (+/- 0.01)
#                 CV SVM_linl1 dev accuracy: 0.66 (+/- 0.16)
#                 CV SVM_linl2 train accuracy: 0.98 (+/- 0.02)
#                 CV SVM_linl2 dev accuracy: 0.64 (+/- 0.17)
# Experiment 4: Same as exp.3 except changed back to dev_size = 42 since the dev errors were better in exp. 1
#     Results:    CV SVM_linl1 train accuracy: 0.98 (+/- 0.01)
#                 CV SVM_linl1 dev accuracy: 0.61 (+/- 0.15)
#                 CV SVM_linl2 train accuracy: 0.98 (+/- 0.01)
#                 CV SVM_linl2 dev accuracy: 0.63 (+/- 0.12)


 #   main()