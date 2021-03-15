import tensorflow as tf

def eve_loss(message, output, batch_size):

    #We map -1 to 0 and +1 to 1 to calculate the loss
    eve_difference = (output +1.0)/2.0 - (message + 1.0)/2.0
    loss = (1/batch_size)*tf.reduce_sum(tf.abs(eve_difference))

    return loss


def alice_bob_loss(message, output, eve_loss, batch_size, text_size):

    #We map -1 to 0 and +1 to 1 to calculate the loss
    alice_bob_difference = (output + 1.0)/2.0 - (message + 1.0)/2.0
    bob_reconstruct_loss = (1/batch_size)*tf.reduce_sum(tf.abs(alice_bob_difference))
    
    # To make sure Eve at least has 50% of bits wrong so that output simulates random probability of binary output
    # ((N/2 - EveLoss)^2)/((N/2)^2)
    eve_evedropping_loss = tf.reduce_sum( tf.square(float(text_size) / 2.0 - eve_loss) / (text_size / 2)**2)
    loss = bob_reconstruct_loss + eve_evedropping_loss

    return loss