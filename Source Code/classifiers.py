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
from sklearn.metrics import roc_curve, auc
from scipy import interp
#decomposed code
import features 
import feature_sel
import dramp
import metrics
import gS
import cV
import test

#plotting parameters
params = {'backend': 'pdf','axes.labelsize': 18,'text.fontsize': 16,'legend.fontsize': 16,'xtick.labelsize': 16,
    'ytick.labelsize': 16, 'xtick.major.pad': 2, 'ytick.major.pad': 2}



### MAIN METHOD: Leave all functions above because it's getting confusing to read the code with "main method"
### commands interspersed throughout script ###

#main methods will exit and not make -i work
#def main():

#read data
CAMP_seqs, CAMP_labels = camp.read_CAMP()
APD_seqs, APD_labels = apd.read_APD('APD_antifungal_165.txt', 'APD_antibacterial_393.txt')
#APD_seqs2, APD_labels2 = apd.read_APD('APD_antifungal_1047.csv', 'APD_antibacterial_2448.csv')

DRAMP_seqs, DRAMP_labels = dramp.seqs_and_labels('dramp_data_antibact_alpha.csv', 'fungal', 3)
#total of 655 sequences
all_seqs = np.concatenate((CAMP_seqs, DRAMP_seqs, APD_seqs))#, APD_seqs2))
all_labels = np.concatenate((CAMP_labels, DRAMP_labels, APD_labels))#, APD_labels2))
#remove redundancies across data sets for a total of 511 unique sequences
unique_seqs, inds = np.unique(all_seqs, return_index=True)
unique_labels = all_labels[inds]
unique_labels[unique_labels == -1.] = 0. ## change -1s to 0
MINLENSEQ = len(min(unique_seqs,key = len)) #shortest peptide in entire database = 6
num_pos = np.sum(unique_labels) #Number of positives (antifungal activity) = 203

#split into positives and negatives; first column contains sequences, second column contains labels
pos_seqs = unique_seqs[unique_labels==1]
pos_labels = unique_labels[unique_labels==1]
pos_examples = np.array([pos_seqs, pos_labels]).T
neg_seqs = unique_seqs[unique_labels==0]
neg_labels = unique_labels[unique_labels==0]
neg_examples = np.array([neg_seqs, neg_labels]).T
#randomly subsample 203 antibacterial AMPs to create a balanced mini data set of 406 sequences
np.random.seed(100)
np.random.shuffle(neg_examples) 
neg_subset = neg_examples[0:num_pos, :]
#combine positive and negative examples for mini data set of 406 examples
# mini_set = np.concatenate((pos_examples, neg_subset), axis=0)
# #compute feautres and labels of entire mini_set
# mini_features, inds = features.constructAllFeatures(mini_set[:,0])
# mini_features = np.array(mini_features)
# #mini_features_z = stats.zscore(mini_features, axis=0, ddof = 1);
# unbalanced_test_addition = neg_examples[num_pos + 8: num_pos + 19, :]
# unbalanced_ta_features, inds = features.constructAllFeatures(unbalanced_test_addition[:,0],minseqlen_ext = 8)
# unbalanced_ta_features = np.array(unbalanced_ta_features)
# unbalanced_ta_labels = unbalanced_test_addition[:,1].astype(float)
# np.save('mini_features_406', mini_features)
mini_features = np.load('mini_features_406_z.npy')

#mini_features_PCA, variance = features.performPCA(mini_features,n_components=227)
#mini_features_tSNE = features.performtSNE(mini_features,n_components=3, perplexity=10)

# mini_labels = mini_set[:,1].astype(float)
#np.save('mini_labels_406', mini_labels)
mini_labels = np.load('mini_labels_406.npy')

#split mini_set into train/dev set and test set according to the above sizes
TEST_SIZE = 40 #~10% of mini_set
# stratify parameter maintains balanced classes according to labels
train_dev_features, test_features, train_dev_labels, test_labels = train_test_split(
                mini_features, mini_labels, test_size=TEST_SIZE, stratify=mini_labels, random_state = 0)
# test_labels = np.concatenate((test_labels, unbalanced_ta_labels))
# test_features = np.concatenate((test_features, unbalanced_ta_features),axis =0)
print 'train_dev_features size: ', train_dev_features.shape
print 'train_dev_labels size: ', train_dev_labels.shape
print 'test_features size: ', test_features.shape
print 'test_labels size: ', test_labels.shape


### MANUAL MODEL TESTING ###

### FEATURE SELECTION ###

TRAIN_SIZE = 292 #80% of train_dev data
DEV_SIZE = 42 #only 11% of train_dev data
#attempt to build SVM with l1 penalty


### CROSS VALIDATED MODEL TESTING ###
SVM_linl1_fs = svm.LinearSVC(C = 1, penalty ='l1', loss = 'squared_hinge', dual = False, random_state=0)
SVM_linl1_fs.fit(train_dev_features,train_dev_labels)
coeff_linl1 = SVM_linl1_fs.coef_
linl1_onerun_features = train_dev_features[:,coeff_linl1[0,:]!=0]
linl1_test_features = test_features[:,coeff_linl1[0,:]!=0]
#PCA_td_features,_, components = features.performPCA(linl1_onerun_features,150)
#PCA_test_features = np.dot(linl1_test_features,components.T)#Use the components matrix to transform test features
PCA_td_features,_, components = features.performPCA(train_dev_features, n_components = 227)
PCA_test_features = np.dot(test_features,components.T)#Use the components matrix to transform test features


print 'Features surviving one run from l1 SVM: ', linl1_onerun_features.shape
#run cross validation on the features resulting from a single run of l1 feature selection
SVM_rbf_m = svm.SVC(kernel = 'rbf')
SVM_linl1_m = svm.LinearSVC( penalty='l1',loss = 'squared_hinge',dual= False)
SVM_linl2_m = svm.LinearSVC(penalty='l2',loss = 'squared_hinge',dual = True)
SVM_poly_m = svm.SVC(kernel = 'poly', degree = 3)
logisticl2_m = linear_model.LogisticRegression() #logreg.LogReg(its = 20000, alpha = .001, Lambda = .25)

#Run cross validation over all features
#SVM_linl1 above
SVM_linl1 = svm.LinearSVC(penalty='l1', loss = 'squared_hinge', dual = False, random_state = 0)
SVM_rbf = svm.SVC(kernel = 'rbf')
SVM_poly = svm.SVC(kernel='poly', degree = 2)
SVM_linl2 = svm.LinearSVC(penalty='l2', loss = 'squared_hinge', dual = True)
logisticl2 = linear_model.LogisticRegression()#logreg.LogReg(its = 5000, alpha = .001, Lambda = .25)
#abandon our crappy logistic regression code for builtin (speed up Grid Search)

#Run cross validation on the features resulting from our optimal set of features. 
#With new code 5 is enough to reduce the number of features considerably without completely killing performance.
inds, bagged_features = feature_sel.cv_featureSelect(SVM_linl1_fs, train_dev_features, train_dev_labels, DEV_SIZE, n=5)
print 'Selected features: ', bagged_features.shape
bagged_test_features=test_features[:,inds]
SVM_rbf_s  = svm.SVC( kernel = 'rbf')
SVM_poly_s = svm.SVC(kernel = 'poly',degree = 3)
SVM_linl1_s =svm.LinearSVC( penalty = 'l1',loss = 'squared_hinge', dual = False)
SVM_linl2_s = svm.LinearSVC(penalty='l2',loss= 'squared_hinge', dual = True)
logisticl2_s = linear_model.LogisticRegression()#logreg.LogReg(its = 40000, alpha = .001, Lambda = .25)

SVM_rbf_PCA  = svm.SVC(C = 1, gamma = .001, kernel = 'rbf')
SVM_poly_PCA = svm.SVC(C = 10, kernel = 'poly',degree = 3)
SVM_linl1_PCA =svm.LinearSVC(C = 100, penalty = 'l1',loss = 'squared_hinge', dual = False)
SVM_linl2_PCA = svm.LinearSVC(C = .1, penalty='l2',loss= 'squared_hinge', dual = True)
logisticl2_PCA = linear_model.LogisticRegression(C = 100000)

classifiers = [SVM_rbf, SVM_poly, SVM_linl1, SVM_linl2, logisticl2]
classifiers_m = [SVM_rbf_m, SVM_poly_m, SVM_linl1_m, SVM_linl2_m, logisticl2_m]
classifiers_s = [SVM_rbf_s, SVM_poly_s, SVM_linl1_s, SVM_linl2_s, logisticl2_s]
classifiers_PCA = [SVM_rbf_PCA, SVM_poly_PCA, SVM_linl1_PCA, SVM_linl2_PCA, logisticl2_PCA]

names = ['SVM_rbf', 'SVM_poly','SVM_lin1', 'SVM_linl2', 'logisticl2']
names_m = ['SVM_rbf_m', 'SVM_poly_m','SVM_lin1_m', 'SVM_linl2_m', 'logisticl2_m']
names_s = ['SVM_rbf_s', 'SVM_poly_s','SVM_lin1_s', 'SVM_linl2_s', 'logisticl2_s']
names_PCA = ['SVM_rbf_PCA', 'SVM_poly_PCA','SVM_lin1_PCA', 'SVM_linl2_PCA', 'logisticl2_PCA']

cv_scores, accstds = cV.crossValidate(train_dev_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
                                models=classifiers, model_names=names)
cv_scores_m, accstds_m = cV.crossValidate(linl1_onerun_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
                                models=classifiers_m, model_names=names_m)
cv_scores_s, accstd_s = cV.crossValidate(bagged_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
                                models=classifiers_s, model_names=names_s)
cv_scores_p, accstd_p = cV.crossValidate(train_dev_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
                                 models=classifiers_PCA, model_names=names_PCA, pca = 1, ncomponents = 227)

scores = test.score(classifiers, names,  train_dev_features,  train_dev_labels, test_features, test_labels)
scores_m = test.score(classifiers_m, names_m, linl1_onerun_features, train_dev_labels, linl1_test_features, test_labels)
scores_s = test.score(classifiers_s, names_s, bagged_features, train_dev_labels, bagged_test_features, test_labels)
scores_PCA = test.score(classifiers_PCA, names_PCA, PCA_td_features,train_dev_labels, PCA_test_features, test_labels)


train_features, dev_features, train_labels, dev_labels = train_test_split(
                 linl1_onerun_features,train_dev_labels, test_size=DEV_SIZE, stratify=train_dev_labels)
y_score = SVM_linl2_m.fit(train_features, train_labels).decision_function(dev_features)

# # Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
# fpr, tpr, _ = roc_curve(dev_labels.reshape(-1,1), y_score)
# roc_auc = auc(fpr, tpr)
average_precision = average_precision_score(dev_labels, y_score)
precision, recall, _ = precision_recall_curve(dev_labels, y_score)
plt.figure(500)
plt.clf()

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('SVM_linl2_m Dev Set PRC : AUC={0:0.2f}'.format(
          average_precision))
plt.savefig("PRC_dev")
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(dev_labels, y_score)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure(2)
plt.clf()
plt.rcParams.update(params)   

lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % (float(roc_auc["micro"])))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM_linl2_m ROC Curve: Dev Set')
plt.legend(loc="lower right")
plt.show()
plt.savefig('ROC_dev')

y_score = SVM_linl2_m.fit(train_dev_features, train_dev_labels).decision_function(test_features)
average_precision = average_precision_score(test_labels, y_score)
precision, recall, _ = precision_recall_curve(test_labels, y_score)
plt.figure()
plt.clf()
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('SVM_linl2_m PRC : AUC={0:0.2f}'.format(
          average_precision))
plt.savefig('PRC_test')          
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
# fpr, tpr, _ = roc_curve(dev_labels.reshape(-1,1), y_score)
# roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels, y_score)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure(3)
plt.clf()
plt.rcParams.update(params)   

lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % (float(roc_auc["micro"])))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM_linl2_m ROC Curve: Test')
plt.legend(loc="lower right")
plt.show()
plt.savefig('ROC_test')


##plot tSNE features against each other to observe distribution
plt.figure(1)
plt.clf()
plt.plot(mini_features_tSNE[mini_labels==1,0], mini_features_tSNE[mini_labels==1,2],'rx',label='positive')
plt.plot(mini_features_tSNE[mini_labels!=1,0], mini_features_tSNE[mini_labels!=1,2],'bo', label='neg')
plt.legend()
plt.title('tSNE feature 0 vs. 2')
plt.xlabel('feature 0')
plt.ylabel('feature 2')
plt.savefig('tSNE0v2.png')

plt.figure(2)
plt.clf()
plt.plot(mini_features_tSNE[mini_labels==1,1], mini_features_tSNE[mini_labels==1,2],'rx',label='positive')
plt.plot(mini_features_tSNE[mini_labels!=1,1], mini_features_tSNE[mini_labels!=1,2],'bo', label='neg')
plt.title('tSNE feature 1 vs. 2')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.savefig('tSNE1v2.png')



Experiment for finding the best number of L1 classifiers to use for feature selection                                
maximum number of splits to try bagging
N = 20
nfeatures = np.zeros(N)
names_subsetted = names
classifiers_subsetted = classifiers
accuracies_stds_over_N = np.zeros((len(names_subsetted),4,N))#4 for train and test accuracy and std
#3d array: row = model, col = metric (tr_acc, tr_std, dev_acc, dev_std), 
#         page = number of shuffles used to come up with features
for it in range(1,N + 1):
    inds, bagged_features = feature_sel.cv_featureSelect(SVM_linl1_fs, train_dev_features, train_dev_labels, DEV_SIZE, n=it)
    print 'Selected features for n = %d: ' % it, bagged_features.shape
    cv_scores_subsetted,accstds = cV.crossValidate(bagged_features, train_dev_labels, DEV_SIZE, n_shuffles = 15, \
                                models=classifiers_subsetted, model_names=names_subsetted, nl1 = it)
    accuracies_stds_over_N[:,:,it-1] = accstds
    nfeatures[it-1] = bagged_features.shape[1]
plt.rcParams.update(params)   
#Plot each model's progress over the number of (folds of feature selection, or number of features?)
for model in range(len(names_subsetted)):
    train_acc_over_N = accuracies_stds_over_N[model,0,:]
    train_std_over_N = accuracies_stds_over_N[model,1,:]
    dev_acc_over_N = accuracies_stds_over_N[model,2,:]
    dev_std_over_N = accuracies_stds_over_N[model,3,:]
    plt.figure(200 + 2*model)
    plt.errorbar(range(1,N+ 1),train_acc_over_N, xerr= None, yerr = train_std_over_N, label = 'train')
    plt.errorbar(range(1,N+ 1),dev_acc_over_N, xerr= None, yerr = dev_std_over_N, label = 'dev')
    plt.title('Mean CV (k = 15) Train/Dev Set Accuracy for %s vs. N_rounds' % names[model])
    plt.xlabel('Number of Classifiers Before Feature Selection')
    plt.legend(loc = 'lower right')
    plt.ylabel('Accuracy')
    plt.savefig('%s.png' % names[model])
    plt.savefig('%s.pdf' % names[model], format = 'pdf')
    plt.figure(200 + 2*model + 1)
    plt.errorbar(nfeatures,train_acc_over_N,xerr=None,yerr=train_std_over_N,label='train')
    plt.errorbar(nfeatures,dev_acc_over_N,xerr=None,yerr=dev_std_over_N,label='dev')
    plt.title('Mean CV (k = 15) Train/Dev Set Accuracy for %s vs. N_features' % names[model])
    plt.xlabel('Number of Features')
    plt.legend(loc = 'lower right')
    plt.ylabel('Accuracy')
    plt.savefig('%s_vsfeatures.png' % names[model])
    plt.savefig('%s_vsfeatures.pdf' % names[model], format = 'pdf')
    
    
MM = metrics.constructMetricsTable(names,cv_scores)
MM_m = metrics.constructMetricsTable(names_m,cv_scores_m)
MM_s = metrics.constructMetricsTable(names_s,cv_scores_s)
MM_pca = metrics.constructMetricsTable(names_PCA, cv_scores_pca)
np.savetxt("Metrics_AllFeatures.csv", MM,fmt = '%s', delimiter=",")
np.savetxt("Metrics_SingleL1.csv", MM_m,fmt = '%s', delimiter=",")
np.savetxt("Metrics_NOpt.csv", MM_s, fmt = '%s',delimiter=",")
np.savetxt("Metrics_PCA.csv", MM_PCA, fmt = '%s',delimiter=",")
MMtest = metrics.constructMetricsTable(names,scores)
MMtest_m = metrics.constructMetricsTable(names_m,scores_m)
MMtest_s = metrics.constructMetricsTable(names_s,scores_s)
MMtest_PCA = metrics.constructMetricsTable(names_PCA,scores_PCA)
np.savetxt("Metrics_PCA.csv", MMtest_PCA,fmt = '%s', delimiter=",")
np.savetxt("Metrics_AllFeaturesTest.csv", MMtest,fmt = '%s', delimiter=",")
np.savetxt("Metrics_SingleL1TEst.csv", MMtest_m,fmt = '%s', delimiter=",")
np.savetxt("Metrics_NOptTest.csv", MMtest_s, fmt = '%s',delimiter=",")


# gS.gridSearch(train_dev_features, train_dev_labels, DEV_SIZE)
#gS.gridSearch(linl1_onerun_features, train_dev_labels, DEV_SIZE)
# gS.gridSearch(bagged_features, train_dev_labels, DEV_SIZE)
# gS.gridSearch(PCA_td_features, train_dev_labels, DEV_SIZE)



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
