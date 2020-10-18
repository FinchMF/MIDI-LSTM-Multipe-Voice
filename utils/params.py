import os

def set_params():

    params = {}
    # file locations
    params['Section'] = 'generative_comps'
    params['run_id'] = '0007'
    params['music_name'] = 'harmony-and-progression'
    # directory structure
    params['run_path'] = 'run/' + params['Section'] + '/'
    params['run_folder'] = params['run_path'] + '_'.join([params['run_id'], params['music_name']])
    params['store_folder'] = os.path.join(params['run_folder'], 'storage')
    params['data_folder'] = os.path.join('data', params['music_name'])
    params['weights_folder'] = os.path.join(params['run_folder'], 'weights')
    params['training_stages'] = os.path.join(params['run_folder'], 'training_stages')
    params['output_folder'] = os.path.join(params['run_folder'], 'output')
    # model params
    params['input_nodes'] = 1024
    params['max_length'] = 15
    params['time_step'] = 1
    # configuration parameters
    params['piano_roll'] = 128
    params['min_value'] = 0.00
    params['lower_first'] = 0.0
    params['lower_second'] = 0.5
    params['upper_first'] = 0.5
    params['upper_second'] = 1.0
    params['max_value'] = 1.0
    params['lower_bound'] = (params['lower_first'] + params['lower_second'] / 2)
    params['upper_bound'] = (params['upper_first'] + params['upper_second'] / 2)
    # contraint parameters
    params['from_where'] = 0
    params['continuation'] = 0.0
    params['first_touch'] = 1.0
    # gen parameters
    params['temperature'] = [0.7, 2.7]
    params['final_temperature'] = [1.2, 0.8]
    # training parameters
    params['EPOCHS_TOTAL'] = 400
    params['EPOCHS'] = 1
    params['BATCH_SIZE'] = 32
    params['VAL_SPLIT'] = 0.2

    return params


def tone_set(origin, end):

    params = {}
    params['origin'] = int(origin)
    params['tones'] = 12
    params['R'] = list(len(params['tones']))
    params['interval'] = int(params['origin'] - 6)
    params['twlv_tn_sorted'] = int(params['R'][params['interval']:] + params['R'][:(params['interval'])%params['tones']])
    params['indx'] = int(params['twlv_tn_sorted'].index(params['end']))
    params['abs_interval'] = abs(6 - (params['indx']))
 
    return params


def chord_set():

    chords = {}
    chords['major'] = [0,4,7]
    chords['minor'] = [0,3,7]
    chords['suspended'] = [0,5,7]
    chords['augmented'] = [0,4,8]
    chords['diminished'] = [0,3,6]
    chords['major_sixth'] = [0,4,7,9]
    chords['minor_sixth'] = [0,3,7,9]
    chords['dominant_seventh'] = [0,4,7,10]
    chords['major_seventh'] = [0,4,7,11]
    chords['minor_sventh'] = [0,3,7,10]
    chords['half_dim_seventh'] = [0,3,6,10]
    chords['dim_seventh'] = [0,3,6,9]
    chords['major_ninth'] = [0,2,4,7,11]
    chords['dominant_ninth'] = [0,2,4,7,10]
    chords['dom_minor_ninth'] = [0,1,4,7,10]
    chords['minor_ninth'] = [0,2,3,7,10]
    

    chord_strings = [
        
        'major',
        'minor',
        'suspended',
        'augmented',
        'diminshed',
        'major_sixth',
        'minor_sixth',
        'dominant_seventh',
        'major_seventh',
        'minor_seventh',
        'half_dim_seventh',
        'dim_seventh',
        'major_ninth',
        'dominant_ninth',
        'dom_minor_ninth',
        'minor_ninth'
    
    ]   

    return chords, chord_strings


class Harmonic_Set:

    def __init__(self, origin, end):

        self.origin = origin
        self.end = end
        self.tone_set = tone_set(self.origin, self.end)
        self.chords, self.chord_strind = chord_set()
        self.flatten = lambda l: [ele for sub in l for ele in sub]
        self.chord_list = self.flatten([[{(n+r)%12 for n in v} for v in self.chords] for r in self.tone_set['R']])
        self.unique_chords = self.unique()

    def unique(self):

        unique_list = []
        
        for idx in range(192):

            if self.chord_list[idx] not in unique_list:
                unique_list.append(self.chord_list[idx])
        
        return unique_list