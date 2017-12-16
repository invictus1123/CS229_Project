### Read in APD data ###

import numpy as np
import propy
import csv

#read in APD data
def read_APD(fungalFile, bacterialFile):
    #read in 165 antifungal AMPs
    APD_data_f = np.loadtxt (fungalFile, dtype = str)
    #for some reason, loadtxt appends '\xca' to each sequence, so remove these characters
    APD_data_f = np.array([s.strip('\xca') for s in APD_data_f])
    f_labels = np.ones(APD_data_f.shape)
    
    #read in 393 antibacterial AMPs
    APD_data_b = np.loadtxt(bacterialFile, dtype = str)
    #for some reason, loadtxt appends '\xca' to each sequence, so remove these characters
    APD_data_b = np.array([s.strip('\xca') for s in APD_data_b])
    #remove sequences in APD_data_b that are in APD_data_f
    # assume that the remaining 239 sequences are only antibacterial, and not antifungal
    APD_b_unique = np.setdiff1d(APD_data_b, APD_data_f)
    b_labels = np.zeros(APD_b_unique.shape)
    #in total there are 404 unique sequences from APD, 165 antifungal, and 239 antibacterial
    APD_seqs = np.concatenate((APD_data_f, APD_b_unique))
    APD_labels = np.concatenate((f_labels, b_labels))
    return APD_seqs, APD_labels

