import tensorflow as tf

def model(collection, message, key=None):

    if key is None:
        # If key is not present, then it's Eve model trying to eavesdrop hence pass the message as input tensor
        comb_in = message
    else:
        # if there is a key then it's either Alice or Bob model trying to encrypt and decrypt resp
        # message and key is concatenated into single tensor and fed as an input
        comb_in = tf.concat(axis = 1, values = [message,key])
    
    # collection arg is to denote the scope of model which is used to aggregate the 
    # tensor to make sure all tensor which needs to be trained are inside one scope
    with tf.variable_scope(collection):
        
        fc = tf.layers.dense(comb_in, t_size + k_size, activation= tf.nn.relu)
        # Convert FC layer a vector of 32 to matrix of (32,1)
        fc = tf.expand_dims(fc, 2)
                
        ## Convolution Layers - Sigmoid activation function
        # input: (32,1) -> output:(32,2) because filter is 2 which creates 2 channels
        conv1 = tf.layers.conv1d( fc,    filters=2, kernel_size=4, strides= 1, padding='SAME',  activation=tf.nn.sigmoid)

        # input: (32,2) -> output:(16,4) because stride is 2 hence result is halved ( i.e 32/2 )
        # filter is 4 which creates 4 channels
        conv2 = tf.layers.conv1d( conv1, filters=4, kernel_size=2, strides= 2, padding='VALID', activation=tf.nn.sigmoid)

        # input: (16,4) -> output:(16,4) because filter is 4 which creates 4 channels
        conv3 = tf.layers.conv1d( conv2, filters=4, kernel_size=1, strides= 1, padding='SAME',  activation=tf.nn.sigmoid)

        ## Convolution Layers - Tanh activation function

        # input: (16,4) -> output:(16,1) because filter is 1 which creates 1 channel
        conv4 = tf.layers.conv1d( conv3, filters=1, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.tanh)

        # Opposite of expand_dims function, here (16,1) tensor is converted to tensor of (16)
        conv4 = tf.squeeze(conv4, 2)
    return conv4