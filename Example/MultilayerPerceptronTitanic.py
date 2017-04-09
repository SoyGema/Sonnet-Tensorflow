from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt 
import pandas as pd 

train_data = get_training_data()
test_data = get_test_data()

def get_training_data():
  data = pd.read_csv('/titanic_train.csv')
  return data 

linear_to_hidden = snt.Linear(output_size = 50 , name ='inp_to_hidden')
hidden_to_out = snt.Linear(output_size = 2, name = 'hidden_to_out')

#Sequential is a module wich applies a number of inner ops in sequence to data 
#We can put another activation function like tf.tahn or tf.nn.relu 
mlp = snt.Sequential([linear_to_hidden, tf.sigmoid, hidden_to_out])

train_predictions = mlp(train_data)
test_predictions = mlp(test_data)
