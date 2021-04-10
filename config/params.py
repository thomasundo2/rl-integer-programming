
# TODO: build a dynamic parameter builder which takes in m and n
dense_params = {'model'      : 'dense',
              'model_params': {'m'          : 20,
                               'n'          : 10,
                               't'          : 10,
                               'lr'         : 0.001}
              }

attention_params = {'model'      : 'attention',
                   'model_params': {'n'          : 10,
                                    'h'          : 6,
                                    'lr'         : 0.001}
                  }

rnn_params = {'model'      : 'rnn',
              'model_params': {'n'          : 10,
                               'lr'         : 0.001}
            }

