import torch

from rnd_network import RNDNetwork, NoRND

def build_rnd(rnd_params):
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rnd_params['model'] == 'dense':
        rnd = RNDNetwork(**rnd_params['model_params']).to(mydevice)
    elif rnd_params['model'] == 'None':
        rnd = NoRND()
    else:
        raise NotImplementedError
    return rnd


