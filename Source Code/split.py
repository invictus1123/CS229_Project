import numpy as np
def balTrainUnbalDevSplit(features, labels, nfolds = 1):
    cvInds = []
    for i in range(nfolds):
        #hard code a ratio of 20/31 for the dev set; balanced for the train set
        pos = features[labels==1]
        neg = features[labels==0]
        posinds = np.argwhere([labels==1])[:,1]
        neginds = np.argwhere([labels==0])[:,1]

        neginds_r = np.random.permutation(len(neginds))
        posinds_r = np.random.permutation(len(posinds))
        original_inds_p = posinds[posinds_r]
        original_inds_n = neginds[neginds_r]

        dev_pos = pos[posinds_r[0:19],:]
        dev_neg = neg[neginds_r[0:31],:]
        dev_inds = np.concatenate((original_inds_p[0:19],original_inds_n[0:30]))
        train_inds = np.concatenate((original_inds_p[20:],original_inds_n[-(len(posinds)-20):]))

        train_pos = pos[posinds_r[20:],:]
        train_neg = neg[-len(train_pos):,:]
        train_features= features[train_inds,:]
        train_labels = labels[train_inds]
        dev_features = features[dev_inds,:]
        dev_labels = labels[dev_inds]
        cvInds.append((train_inds, dev_inds))
        #for use in crossValidate: perform this nfolds times and append the indices
    return train_features,dev_features,train_labels, dev_labels, cvInds