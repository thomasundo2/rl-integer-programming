
# TODO: build a dynamic parameter builder which takes in m, n, and t
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


attention_params = {'model'      : 'attention',
                   'model_params': {'n'          : 60,
                                    'h'          : 40,
                                    'lr'         : 0.005}
                  }


dense_params = {'model'      : 'dense',
              'model_params': {'m'          : 60,
                               'n'          : 60,
                               't'          : 50,
                               'lr'         : 0.001}
              }



rand_params = {'model'      : 'random',
              'model_params': { }
             }
