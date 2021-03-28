import tensorflow as tf
import numpy as np

def random_bools(size, n):
    #we create a n*size array with values 1, -1
    temp =  np.random.random_integers(low=0, high=1, size=[size, n])
    temp = temp*2 - 1
    
    return temp.astype(np.float32)
  
def get_dataset(sample_size, text_size, key_size):

    message = random_bools(sample_size, text_size)
    key = random_bools(sample_size, key_size)
  
    return message, key

def get_data(batch_size, text_size, key_size):

    #message = tf.placeholder( tf.float32, shape=(batch_size,text_size), name = 'message')
    #key = tf.placeholder(tf.float32, shape=(batch_size, key_size), name='key')

    message = tf.Variable(tf.random.normal(shape=(batch_size,text_size), name = 'message'))
    key     = tf.Variable(tf.random.normal(shape=(batch_size, key_size), name='key'))

    return message, key