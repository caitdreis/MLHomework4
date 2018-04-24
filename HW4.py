#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:05:53 2018

@author: caitdreisbach
"""

import numpy as np
np.random.seed(42)
import tensorflow as tf
from tensorflow.contrib import rnn
tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import _pickle as pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams
from nltk import FreqDist
from os import path
import re



#RAW_PATH = "C:\\Users\\mkw5c\\Documents\\MLHomework4\\down_sampled_reviews\\train_tiny.txt"
#PROCESSED_PATH = "C:\\Users\\mkw5c\\Documents\\School- spring 2018\\Machine Learning\\hw 4 ML\\"

RAW_PATH = "/scratch/cnd2y/train_tiny.txt"
PROCESSED_PATH = "/scratch/cnd2y/"

#RAW_PATH = "/Users/eihoman/Desktop/DSI/Machine Learning/hw/train.txt"
#PROCESSED_PATH = "/Users/eihoman/Desktop/DSI/Machine Learning/hw/"

vocab = []

# load raw corpus and form raw dictionary
print("Loading raw data....")
with open(path.join(RAW_PATH)) as f:
    for line in f:
        line = line.strip().split()
        all_words = [word.lower() for word in line]
        words_cleaner = [re.sub(r'[\W_]+', '', word) for word in all_words]
        words = ["num" if re.search('[0-9]+', word) else word for word in words_cleaner]
        vocab.extend(words)
     
print("%d words loaded" % len(vocab))

# count occurrence of each word
word_freq = FreqDist(vocab)
print("%d unique words" % len(word_freq.items()))

#input vocab
dictionary = word_freq.most_common()
expression = "The least frequent word is '%s', and it occurs %d times" 
print(expression % (dictionary[-1][0], dictionary[-1][1]))
unique_vocab = [x[0] for x in dictionary]
unique_vocab.append("UNKNOWN")
# output vocab
#dictionary = word_freq.most_common(_ONEHOT_SIZE-1)

#expression = "The least frequent word for output is '%s', and it occurs %d times" 
#print(expression % (dictionary[-1][0], dictionary[-1][1]))

#onehot = [x[0] for x in dictionary]
#onehot.insert(0, "UNK")

vocab_size = len(unique_vocab)
vocab_size

#give each word a numeric encoding

word_indices = dict((c, i) for i, c in enumerate(unique_vocab))
indices_word = dict((i, c) for i, c in enumerate(unique_vocab))

print(word_indices['it'])
print(indices_word[8])

encoded_words = [word_indices[word] for word in vocab]

print(encoded_words[0:5])

SEQUENCE_LENGTH = 2
step = 1
X_pairs = []
y_pairs = []
for i in range(0, len(encoded_words) - SEQUENCE_LENGTH, step):
    X_pairs.append(encoded_words[i: i + SEQUENCE_LENGTH])
    y_pairs.append(encoded_words[i+1: i + 1 + SEQUENCE_LENGTH])
print(f'num training examples: {len(X_pairs)}')


#view the last observation
print(X_pairs[0])
print(y_pairs[0])

# convert X_pairs to a numpy array
word_array = np.asarray(X_pairs)
word_array[0]

batch_size = word_array.shape[0]
batch_size

# reshape to have the correct dimensions
word_array2 = word_array.reshape([batch_size, 2, 1])
word_array2[0]

# convert y_pairs to a numpy array
y_array = np.asarray(y_pairs)
y_array.shape

#rename word_array2

X_array = np.copy(word_array2)
print(X_array.shape)
print(X_array[1])

X_subset = X_array[0:100]
y_subset = y_array[0:100]

print(X_subset.shape)
print(y_subset.shape)

#V_RAW_PATH = "C:\\Users\\mkw5c\\Documents\\MLHomework4\\down_sampled_reviews\\valid_tiny.txt"
#PROCESSED_PATH = "C:\\Users\\mkw5c\\Documents\\School- spring 2018\\Machine Learning\\hw 4 ML\\"

V_RAW_PATH = "/scratch/cnd2y/valid_tiny.txt"
PROCESSED_PATH = "/scratch/cnd2y/"

#RAW_PATH = "/Users/eihoman/Desktop/DSI/Machine Learning/hw/train.txt"
#PROCESSED_PATH = "/Users/eihoman/Desktop/DSI/Machine Learning/hw/"

valid_set = []

# load raw corpus and form raw dictionary
print("Loading raw data....")
with open(path.join(V_RAW_PATH)) as f:
    for line in f:
        line = line.strip().split()
        all_words = [word.lower() for word in line]
        words_cleaner = [re.sub(r'[\W_]+', '', word) for word in all_words]
        words = ["num" if re.search('[0-9]+', word) else word for word in words_cleaner]
        words2 = ["UNKNOWN" if word not in unique_vocab else word for word in words]
        valid_set.extend(words2)
     
print("%d words loaded" % len(vocab))

# count occurrence of each word
v_word_freq = FreqDist(valid_set)
print("%d unique words" % len(v_word_freq.items()))

#input vocab
v_dictionary = v_word_freq.most_common()
expression = "The least frequent word is '%s', and it occurs %d times" 
print(expression % (v_dictionary[-1][0], v_dictionary[-1][1]))
valid_vocab = [x[0] for x in dictionary]

# just to check that it is the same size or smaller than the training set
len(valid_vocab)

encoded_words = [word_indices[word] for word in valid_set]
print(valid_set[0:5])
print(encoded_words[0:5])

SEQUENCE_LENGTH = 2
step = 1
X_valid_pairs = []
y_valid_pairs = []
for i in range(0, len(encoded_words) - SEQUENCE_LENGTH, step):
    X_valid_pairs.append(encoded_words[i: i + SEQUENCE_LENGTH])
    y_valid_pairs.append(encoded_words[i+1: i + 1 + SEQUENCE_LENGTH])
print(f'num training examples: {len(X_valid_pairs)}')

#view the last observation
print(X_valid_pairs[0])
print(y_valid_pairs[0])

# convert X_pairs to a numpy array
X_test_array = np.asarray(X_valid_pairs)
X_test_array[0]

batch_size = X_test_array.shape[0]
batch_size

# reshape to have the correct dimensions
X_test_array = X_test_array.reshape([batch_size, 2, 1])
print(X_test_array[0])
print(X_test_array.shape)

# convert y_pairs to a numpy array
y_test_array = np.asarray(y_valid_pairs)
print(y_test_array[0])
print(y_test_array.shape)

import timeit
# Encapsulate the entire prediction problem as a function
def build_and_predict(trainX,trainY,testX,cell,cellType,input_dim=1,hidden_dim=100,seq_size = 2,max_itr=200):


    # Build computational graph
    graph = tf.Graph()

    with graph.as_default():
        # input place holders
        # input Shape: [# training examples, sequence length, input dimensions=1]
        x = tf.placeholder(tf.float32,[None,seq_size,input_dim])
        # label Shape: [# training examples, sequence length]
        y = tf.placeholder(tf.int32,[None,seq_size])

        #get batch_size
        num_obs = tf.shape(x)[0]
        print(num_obs)

        # RNN output Shape: [# training examples, sequence length, # hidden] 
        outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        # Inputs to Fully Connected Layer
                # inputs shape: [batch_size*seq_size, hidden]
        FC_inputs = tf.reshape(outputs, [num_obs*seq_size, hidden_dim])
        print(FC_inputs.shape)
                # W shape: [# hidden, vocab_size]
        W_out = tf.Variable(tf.random_normal([hidden_dim, vocab_size]),name="w_out") 
        #print(W_out.shape)
                # b shape: [vocab_size]
        b_out = tf.Variable(tf.random_normal([vocab_size]),name="b_out")

        print(W_out)

        # output dense layer:
        y_pred = tf.matmul(FC_inputs,W_out)+b_out

        #reshape logits for cost function
        y_pred = tf.reshape(y_pred, [num_obs, seq_size, vocab_size])


        # Cost & Training Step
        cost = tf.contrib.seq2seq.sequence_loss(y_pred, y, W_out, average_across_timesteps = True, average_across_batch = True)
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        
    #--- RUN SESSION 
    with tf.Session(graph=graph) as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # time for comparison to other models
        start=timeit.default_timer()

        # Run for 1000 iterations (1000 is arbitrary, need a validation set to tune!)
        print('Training...')
        for i in range(1000): # If we train more, would we overfit? Try 10000
            o_v=sess.run(outputs,feed_dict={x:trainX, y:trainY})
            print(o_v.shape)
            break
            _, train_err = sess.run([train_op,cost],feed_dict={x:trainX, y:trainY})
            if i==0:
                print('  step, train err= %6d: %8.5f' % (0,train_err)) 
            elif  (i+1) % 100 == 0: 
                print('  step, train err= %6d: %8.5f' % (i+1,train_err)) 

        end=timeit.default_timer()    
        print("Training time : %10.5f"%(end-start))

        # Test trained model 
            # on training data
        predicted_vals_all_train= sess.run(y_pred,feed_dict={x:trainX}) 
            # on validation set
        predicted_vals_all_test= sess.run(y_pred,feed_dict={x:testX}) 
        # Get last item in each predicted sequence:
            # training
        predicted_vals_train = predicted_vals_all_train[:,seq_size-1]
            #testing
        predicted_vals_test = predicted_vals_all_test[:,seq_size-1]

    return predicted_vals

input_dim=1 # dim > 1 for multivariate time series
hidden_dim=100 # number of hiddent units h
max_itr=2000 # number of training iterations
seq_size = 2

# Different RNN Cell Types
RNNcell = rnn.BasicRNNCell(hidden_dim)
LSTMcell = rnn.BasicLSTMCell(hidden_dim)
GRUcell = rnn.GRUCell(hidden_dim)

# Build models and predict on testing data
predicted_vals_rnn=build_and_predict(X_array,y_array, X_test_array,RNNcell,"RNN",input_dim,hidden_dim,seq_size,max_itr)
predicted_vals_lstm=build_and_predict(X_array,y_array, X_test_array,LSTMcell,"LSTM",input_dim,hidden_dim,seq_size,max_itr)
predicted_vals_gru=build_and_predict(X_array,y_array, X_test_array,GRUcell,"GPU",input_dim,hidden_dim,seq_size,max_itr)

# Compute Mean Cross Entropy
def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
    return ce

mce_rnn = cross_entropy(predicted_vals_rnn, y_array)
mce_lstm = cross_entropy(predicted_vals_lstm, y_array)
mce_gru = cross_entropy(predicted_vals_gru, y_array)

print("RNN MCE = %10.5f"%mce_rnn)
print("LSTM MCE = %10.5f"%mce_lstm)
print("GRU MCE = %10.5f"%mce_gru)

# Plot predictions
pred_len=len(predicted_vals_dnorm_rnn)
train_len=len(X_array)

plt.figure()
plt.plot(list(range(seq_size+train_len,seq_size+train_len+train_len)), predicted_vals_dnorm_rnn, color='r', label='RNN')
plt.plot(list(range(seq_size+train_len,seq_size+train_len+train_len)), predicted_vals_dnorm_lstm, color='b', label='LSTM')
plt.plot(list(range(seq_size+train_len,seq_size+train_len+train_len)), predicted_vals_dnorm_gru, color='y', label='GRU')
plt.plot(list(range(len(dataset))), dataset, color='g', label='Actual')
plt.legend()






