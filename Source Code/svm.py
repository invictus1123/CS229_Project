#Taken from pset 2 starter code
import numpy as np
import matplotlib.pyplot as plt
import subprocess

np.random.seed(123)
tau = 8.

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    category = (np.array(Y) * 2) - 1
    return matrix, tokens, category

def svm_train(matrix, category):
    state = {}
    M, N = matrix.shape
    #####################
    Y = category
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(matrix.T)
    K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (tau ** 2)) )

    alpha = np.zeros(M)
    alpha_avg = np.zeros(M)
    L = 1. / (64 * M)
    outer_loops = 40

    alpha_avg
    for ii in xrange(outer_loops * M):
        i = int(np.random.rand() * M)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = M * L * K[:, i] * alpha[i]
        if (margin < 1):
            grad -=  Y[i] * K[:, i]
        alpha -=  grad / np.sqrt(ii + 1)
        alpha_avg += alpha

    alpha_avg /= (ii + 1) * M

    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = matrix
    state['Sqtrain'] = squared
    ####################
    return state

def svm_test(matrix, state):
    M, N = matrix.shape
    output = np.zeros(M)
    ###################
    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (tau ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = np.sign(preds)
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error
    return error

def main():
    #load in all the sizes of training sets
    size = [50, 100, 200, 400, 800, 1400]
    test_errors = []
    #try on different sized training sets
    for x in size:
        filename = 'MATRIX.TRAIN.%s' % x
        print filename
        trainMatrix, tokenlist, trainCategory = readMatrix(filename)
        testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
        state = svm_train(trainMatrix, trainCategory)
        output = svm_test(testMatrix, state)

        test_errors.append(evaluate(output, testCategory))

    #plot test error vs. training set size
    plt.figure()    
    plt.plot(size, test_errors)
    plt.xlabel('Training Set Size')
    plt.ylabel('Test Error')
    plt.savefig('test_error_vs_size_SVM_seed.pdf')
    subprocess.Popen(['open', 'test_error_vs_size_SVM_seed.pdf']).wait()
    return

if __name__ == '__main__':
    main()
