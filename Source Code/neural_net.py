#from workspace, type following command to activate tensorflow virtual environmentsource:
# ./tensorflow/bin/activate

import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

learning_rate = 0.001 #for gradient descent
training_epochs = 100
beta = 0.01 #strength of regularizer
n_hidden = 1 # the number of "neurons" in the first hidden layer of the neural network.

TRAIN_SIZE = 324 #90% of train_dev data
DEV_SIZE = 42 #only 11% of train_dev data
TEST_SIZE = 40 #~10% of mini_set

#Load Data
#compute feautres and labels of entire mini_set
mini_features = np.load('mini_features.npy') 
mini_labels = np.load('mini_labels.npy')
mini_labels[mini_labels == -1.] = 0.

#split mini_set into train/dev set and test set according to the above sizes
# stratify parameter maintains balanced classes according to labels
train_dev_features, test_features, train_dev_labels, test_labels = train_test_split(
                mini_features, mini_labels, test_size=TEST_SIZE, stratify=mini_labels, random_state = 0)

#Feature selection
SVM_linl1_165 = svm.LinearSVC(penalty ='l1', loss = 'squared_hinge', dual = False, random_state=0)
SVM_linl1_165.fit(train_dev_features,train_dev_labels)
coeff_linl1 = SVM_linl1_165.coef_
linl1_onerun_features = train_dev_features[:,coeff_linl1[0,:]!=0]
print 'Features surviving one run from l1 SVM: ', linl1_onerun_features.shape

num_features = linl1_onerun_features.shape[1]

# split train/dev set
train_features, dev_features, train_labels, dev_labels = train_test_split(
                linl1_onerun_features, train_dev_labels, test_size=DEV_SIZE, stratify=train_dev_labels, random_state = 0)              

# tf Graph Input
x = tf.placeholder(tf.float32, [None, num_features]) # one sequence of shape 1400
y = tf.placeholder(tf.float32, [None, 1]) # 2 classes: 1 is antifungal, 0 is antibacterial

# Set model weights/parameters
W1 = tf.get_variable("W1", shape=[num_features, n_hidden],
       initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([n_hidden]))
W2 = tf.get_variable("W2", shape=[n_hidden, 1],
       initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.zeros([1]))

# Construct model

a1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1) # activate the hidden layer
prob_y1 = tf.nn.sigmoid(tf.matmul(a1, W2) + b2) # softmax with 2 dimensions is just a sigmoid
prob_y0 = 1 - prob_y1

#average cross-entropy loss
loss = - tf.reduce_mean(y*tf.log(prob_y1)+(1-y)*tf.log(prob_y0))

# Loss function using L2 Regularization
regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
loss = tf.reduce_mean(loss + beta * regularizers)
    
#Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()

# Start training
# Run the initializer
sess.run(init)
    
xs, ys = train_features, train_labels.reshape(TRAIN_SIZE, 1)
xdev, ydev = dev_features, dev_labels.reshape(DEV_SIZE,1)
train_loss = []
for epoch in range(training_epochs):
    # Run optimization op (backprop) and cost op (to get loss value)
    _, ce_loss = sess.run([optimizer, loss], feed_dict={x: xs, y: ys}) 
    # following https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf
    train_loss.append(ce_loss)
    #Classification accuracy
    classifications = tf.round(prob_y1) # get classifications of each datapoint from probabilities
    accurracy = 1.0 - tf.reduce_mean(tf.abs(classifications - y)) # compute accuracy using classifications and truth y
    training_acc = sess.run(accurracy, feed_dict={x: xs, y: ys})
    dev_acc = sess.run(accurracy, feed_dict={x: xdev, y: ydev})
    # Print train loss, train accuracy, and dev accuracy after each epoch
    print ('Epoch: %d, Train loss: %.4f, Train_acc: %.4f, Dev_acc: %.4f' % 
                (epoch+1, ce_loss, training_acc, dev_acc))
