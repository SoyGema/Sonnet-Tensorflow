from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt 

#------------------------------------------------------------------------------------------------------------------------------#
"""Flags are a TensorFlow internal util to define command-line parameters."""
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'hidden_size', 'output_size'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")

flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

flags.DEFINE_float("hidden_size", 50, "output size for linear model")
flags.DEFINE_float("output_size", 25, "output size for hidden layer")
#------------------------------------------------------------------------------------------------------------------------------#

train_data =
test_data =

linear_to_hidden = snt.Linear(output_size = 50 , name ='inp_to_hidden')
hidden_to_out = snt.Linear(output_size = 25, name = 'hidden_to_out')

#Sequential is a module wich applies a number of inner ops in sequence to data 
#We can put another activation function like tf.tahn or tf.nn.relu 
mlp = snt.Sequential([linear_to_hidden, tf.sigmoid, hidden_to_out])

train_predictions = mlp(train_data)
test_predictions = mlp(test_data)
