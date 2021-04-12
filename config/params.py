
# CONFIG FOR STARTER
"""
dense_params = {'model'      : 'dense',
              'model_params': {'m'          : 20,
                               'n'          : 10,
                               't'          : 10,
                               'lr'         : 0.001}
              }

attention_params = {'model'      : 'attention',
                   'model_params': {'n'          : 10,
                                    'h'          : 6,
                                    'lr'         : 0.005}
                  }

# todo: fix parallelization for rnns
rnn_params = {'model'      : 'rnn',
              'model_params': {'n'          : 10,
                               'lr'         : 0.001}
            }

rand_params = {'model'      : 'random',
              'model_params': { }
             }
"""

def attention_params(n, h, lr):
    return {'model'       : 'attention',
            'model_params': {'n'          : n,
                             'h'          : h,
                             'lr'         : lr}
           }

def dense_params(m, n, t, lr):
    return {'model'       : 'dense',
            'model_params': {'m'          : m,
                             'n'          : n,
                             't'          : t,
                             'lr'         : lr}
          }

def rand_params():
    return {'model'       : 'random',
            'model_params': { }
           }

