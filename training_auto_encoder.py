import scipy
import sklearn
import os

print()
from sklearn.externals import joblib

tfidf_vectorizer = joblib.load('/home/deepak/DocumentClusteringUsingSelfLearningNetwork/tfidf_matrix.pkl')

inputArray = tfidf_vectorizer.toarray();

#Network Parameters

n_input = inputArray.shape[1]

n_hidden_1 = int(n_input/3)

n_hidden_2 = int(n_hidden_1/3)

n_hidden_3 = int(n_hidden_2/3)


Training_Data = inputArray[400:]

Test_Data = inputArray[:100]


print("Test")

import tensorflow as tf


Test_Tensor = tf.placeholder(dtype='float32',shape=n_input)


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
}


# Parameters
learning_rate = 0.01
training_epochs = 8
batch_size = 50
display_step = 1
