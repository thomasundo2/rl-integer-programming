from rnd_network import RNDNetwork, NoRND

def build_rnd(rnd_params):
    if rnd_params['model'] == 'dense':
        rnd = RNDNetwork(**rnd_params['model_params'])
    elif rnd_params['model'] == 'None':
        rnd = NoRND()
    else:
        raise NotImplementedError
    return rnd


