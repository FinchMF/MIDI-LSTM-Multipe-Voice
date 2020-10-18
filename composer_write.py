

import sys
import subprocess
import random
from utils import params as par

from NN import composer_utils
from NN import composer_rnn

from composer_train import train_RNN
from keras.models import load_model


def compose(compsr):

    params = par.set_params()

    try: 
        net = load_model(f'./models/{compsr}_piano_model.h5')

    except:
        print('[+] Model Does Not Exist..')
        print('[+] Initiate Network Training...')
        net = train_RNN(compsr)

    midi_arr = composer_utils.build_midi_database(str(compsr), params)
    start_idx = random.randint(0, len(midi_arr) - params['max_length'] - 1)
    gen_midi = midi_arr[start_idx: start_idx + params['max_length']]
    
    composer_utils.generate_midi(net, midi_arr, start_idx, gen_midi, params, e='final_iter')

    print('[+] Generated Midi saved')

    return None

if __name__ == '__main__':

    composer_name = sys.argv[1]
    compose(composer_name)



    



