### Read in CAMP data ###

import numpy as np
import propy
import csv
from scipy import stats

#read in CAMP data
def read_CAMP():
    CAMP_data_f= []
    #read in 30 antifungal AMPs
    with open('CAMPSTR_antifungal_helical_30.csv') as csvfile:
        readCSV= csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            CAMP_data_f.append(row)
    CAMP_data_f = np.array(CAMP_data_f)

    #append 85 antibacterial AMPs
    CAMP_data_b = []
    with open('CAMPSTR_antibacterial_helical_85_alt.csv') as csvfile:
        readCSV= csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            CAMP_data_b.append(row)
    CAMP_data_b = np.array(CAMP_data_b)

    CAMP_data = np.concatenate((CAMP_data_f[1:, :], CAMP_data_b[1:, :]), axis=0)
    ### PRE-PROCESS DATA ###
    #only want sequence, and activity
    CAMP_data= CAMP_data[:, [5, 9]]
    #subset the unique examples -- in total, 74
    CAMP_seqs_unique, indices= np.unique(CAMP_data[:,0], return_index = True)
    CAMP_data_unique = CAMP_data[indices,:]
    #replace activity label with 1 if antifungal, 0 if antibacterial
    # 25 are antifungal and 49 are antibacterial
    annotation= 'fungal'
    CAMP_labels = np.array([annotation in CAMP_data_unique[i,1] for i in range(len(CAMP_seqs_unique))]).astype(int)
    return CAMP_seqs_unique, CAMP_labels