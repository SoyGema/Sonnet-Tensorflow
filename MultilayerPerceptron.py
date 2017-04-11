from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt 
import numpy as np 
#------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------#

train_data = tf.convert_to_tensor(np.matrix([[1.0,2.0],[3.0,4.0]]), dtype=tf.float32)
test_data = tf.convert_to_tensor(np.matrix([[5.0,6.0],[8.0,9.0]]), dtype=tf.float32)

linear_to_hidden = snt.Linear(output_size = 4, name ='inp_to_hidden')
hidden_to_out = snt.Linear(output_size = 2, name = 'hidden_to_out')

#Sequential is a module wich applies a number of inner ops in sequence to data 
#We can put another activation function like tf.tahn or tf.nn.relu 
mlp = snt.Sequential([linear_to_hidden, tf.sigmoid, hidden_to_out])

train_predictions = mlp(train_data)
test_predictions = mlp(test_data)
