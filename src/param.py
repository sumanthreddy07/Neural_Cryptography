class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_parameters():
    # Input configuration.
    parameters ={
    'text_size'       : 16,
    'key_size'        : 16,

    # Training parameters.
    'learning_rate'   : 0.0008,
    'batch_size'      : 4096,
    'sample_size'     : 4096*5,
    'epochs'          : 850000,

    'iterations'      : 1,
    'eve_multiplier'  : 2  # Train Eve 2x for every step of Alice/Bob
    }
    parameters['steps_per_epoch']=int(parameters['sample_size']/parameters['batch_size'])
    return parameters