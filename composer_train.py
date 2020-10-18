
import sys
import subprocess
import random
import numpy as np 

from NN import composer_utils
from NN import composer_rnn

from utils import params as par   


def train_RNN(compsr):

    params = par.set_params()
    print('\n [+] Parameters Set \n')

    midi_arr = composer_utils.build_midi_database(str(compsr), params)
    print('\n [+] Midi Transformed \n')

    n_samples, max_length, freq, x_axis, preds = composer_utils.fetch_inputs(midi_arr, params)
    print('\n [+] Inputs Set')

    net = composer_rnn.RNN_(params).construct_network()
    print('\n [+] Network Instantiated \n')


    print(f'\n [+] Begin Training...\n')
    for e in range(1, params['EPOCHS_TOTAL']):

        print(f'CURRENT EPOCH: {e}')

        net.fit(x_axis, preds, 
                batch_size=params['BATCH_SIZE'], 
                epochs=params['EPOCHS'],
                validation_split=params['VAL_SPLIT'],
                shuffle=True)

        start_idx = random.randint(0, len(midi_arr) - params['max_length']-1)
        gen_midi = midi_arr[start_idx : start_idx + params['max_length']]
        
        if ((e%10) == 0):

            net.save_weights(params['training_stages'] + f'/{compsr}_weights_{e}_.h5')
            composer_utils.generate_midi(net, midi_arr, start_idx, gen_midi, params, e)

            print(f' EPOCH {e}: Weights Saved - Midi Generated')

        net.save(f'./models/{compsr}_piano_model.h5')

    return net



if __name__ == '__main__':

    composer_name = sys.argv[1]
    train_RNN(composer_name)














