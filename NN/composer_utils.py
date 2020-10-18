

import os
import zipfile
import glob
from pathlib import Path
import numpy as np 
import music21
from music21 import converter, instrument, note, chord, duration, stream


def note_to_int(note):

    def note_value(note, note_base_name, n):
        first_pitch = note[0]
        base_value = note_base_name.index(first_pitch)
        octave = note[int(n)]
        value = base_value + 12*(int(octave)-(-1))
        return value

    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    if ('#-' in note):
        value = note_value(note, note_base_name, 3)
    
    elif ('#' in note):
        value = note_value(note, note_base_name, 2)

    elif ('-' in note):
        value = note_value(note, note_base_name, 2)
    
    else:
        value = note_value(note, note_base_name, 1)

    return value

def notes_to_matrix(notes, durations, offsets, params):

    try:
        last_offset = int(offsets[-1])
    except IndexError:
        print('[!] Index Error')
        return (None, None, None)

    total_offset_axis =  last_offset * 4 + (8 * 4)
    matrix = np.random.uniform(params['min_value'], params['lower_first'], (params['piano_roll'], int(total_offset_axis)))
    
    for (note, duration, offset) in zip(notes, durations, offsets):

        dur = int(float(duration)/0.25)
        first_touch = np.random.uniform(params['upper_second'], params['max_value'], 1)
        continuation = np.random.uniform(params['lower_second'], params['upper_first'], 1)

        if ('.' not in str(note)):
            matrix[note, int(offset * 4)] = first_touch
            matrix[note, int((offset * 4) + 1) : int((offset * 4) + dur)] = continuation

        else:
            chord_notes_str = [note for note in note.split('.')]
            chord_notes_float = list(map(int, chord_notes_str))

            for chord_note_float in chord_notes_float:
                matrix[chord_note_float, int(offset * 4)] = first_touch
                matrix[chord_note_float, int((offset * 4) +1) : int((offset * 4) + dur)] = continuation

    return matrix

def check_float(duration, params):

    if ('/' in duration):
        numerator = float(duration.split('/')[0])
        denominator = float(duration.split('/')[0])
        duration = str(float(numerator/denominator))

    return duration

def midi_to_matrix(filename, params, length=250):

    try:
        midi = converter.parse(filename)
    except UnicodeDecodeError:
        print('[!] Decode Error')
        return None

    notes_to_parse = None
    
    try:
        parts = instrument.partitionByInstrument(midi)
    except TypeError:
        print('[!] Type Error')
        return None

    instrument_names = []

    try:
        for inst in parts:
            name = (str(inst).split(' ')[-1])[:-1]
            instrument_names.append(name)
    except TypeError:
        print('[!] Type is not iterable')
        return None

    try:
        piano_idx = instrument_names.index('Piano')
    except ValueError:
        print(f'[!] {filename} have no Piano part')
        return None

    notes_to_parse = parts.parts[piano_idx].recurse()
    duration_piano = float(check_float((str(notes_to_parse._getDuration()).split(' ')[-1])[:-1], params))
    durations, notes, offsets = [], [], []

    for ele in notes_to_parse:

        if isinstance(ele, note.Note):
            notes.append(note_to_int(str(ele.pitch)))
            duration = str(ele.duration)[27:-1]
            durations.append(check_float(duration, params))
            offsets.append(ele.offset)

        elif isinstance(ele, chord.Chord):
            notes.append('.'.join(str(note_to_int(str(n))) for n in ele.pitches))
            duration = str(ele.duration)[27:-1]
            durations.append(check_float(duration, params))
            offsets.append(ele.offset)

    matrix = notes_to_matrix(notes, durations, offsets, params)

    try:
        freq, time = matrix.shape
    except AttributeError:
        print('[!] -tuple- object has no attribute -shape-')
        return None

    if (time >= length):
        return (matrix[:,:length])
    else:
        print(f'[!] {filename} duration too short')

def int_to_note(integer):

    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave_detector = (integer // 12)
    base_name_detector = (integer % 12)
    note = note_base_name[base_name_detector] + str((int(octave_detector))-1)

    if ('-' in note):
        note = note_base_name[base_name_detector] + str(0)
        return note
    return note

def converter_func(arr, params):

    np.place(arr, arr < params['lower_bound'], -1.0)
    np.place(arr, (params['lower_bound'] <= arr) & (arr < params['upper_bound']), 0.0)
    np.place(arr, arr >= params['upper_bound'], 1.0)

    return arr

def count_repetitions(arr, from_where, continuation):

    new_arr = arr[from_where:]
    count_repetitive = 1

    for idx in new_arr:
        if (idx != continuation):
            return (count_repetitive)
        else:
            count_repetitive += 1

    return (count_repetitive)

def matrix_to_midi(matrix, params, random=0):

    y_axis, x_axis = matrix.shape
    output_notes = []
    offset = 0

    start_zeros = 0
    for x_num in range(x_axis):

        time_interval = matrix[:,x_num]
        time_interval_norm = converter_func(time_interval, params)

        if (params['first_touch'] not in time_interval_norm):

            start_zeros +=1
        else:
            break

    end_zeros = 0
    for x_num in range(x_axis-1, 0, -1):

        time_interval = matrix[:,x_num]
        time_interval_norm = converter_func(time_interval, params)

        if (params['first_touch'] not in time_interval_norm):

            end_zeros +=1
        else:
            break

    print(f'[i] How non-start note at beginning: {start_zeros}')
    print(f'[i] How non-start note at end: {end_zeros}')

    maxtrix = matrix[:, start_zeros:]
    y_axis, x_axis = matrix.shape
    print(f'[i] Shape: {(x_axis, y_axis)}')

    for y_num in range(y_axis):

        freq_interval = matrix[y_num,:]
        freq_interval_norm = converter_func(freq_interval, params)

        i = 0
        offset = 0

        if (random):

            while (i < len(freq_interval)):

                repetitions = 0
                temp_i = i

                if (freq_interval_norm[i] == params['first_touch']):

                    repetitions = count_repetitions(freq_interval_norm, from_where=i+1, continuation=params['continuation'])
                    i += repetitions

                if (repetitions > 0):

                    rand_num = np.random.randint(3, 6)
                    new_note = note.Note(int_to_note(y_num), duration=duration.Duration(0.25*rand_num*repetitions))
                    new_note.offset = 0.25*temp_i*2
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
                else:

                    i +=1

        else:

            while (i < len(freq_interval)):

                repetitions = 0
                temp_i = i

                if (freq_interval_norm[i] == params['first_touch']):

                    repetitions = count_repetitions(freq_interval_norm, from_where=i+1, continuation=params['continuation'])
                    i += repetitions

                if (repetitions > 0):

                    new_note = note.Note(int_to_note(y_num), duration=duration.Duration(0.25*repetitions))
                    new_note.offset = 0.25*temp_i
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)

                else:

                    i +=1

    return output_notes

def build_midi_database(composer, params):

    db_npy = f'{composer}_midi_db.npy'
    db_npy_path = f'data/compose/{composer}/{db_npy}'

    if os.path.exists(db_npy_path):

        midi_arr = np.load(db_npy_path)
    else:
        print(f'[+] {os.getcwd()}')
        collect_midi = glob.glob(f'data/compose/{composer}/*.mid')
        print('[+] Collected Midi')

        all_midi_matrix = []

        for midi in collect_midi:

            print(f'\n[i] {midi}')
            try:
                single_midi_matrix = midi_to_matrix(midi, params, length=300)
            except UnicodeDecodeError:
                print('[!] Unicode Error')
                continue

            if (single_midi_matrix is not None):
                all_midi_matrix.append(single_midi_matrix)
                print(f'[i] Midi Matrix Shape: {(single_midi_matrix.shape)}\n')

        midi_arr = np.asarray(all_midi_matrix)
        print(f'[+] Pre-Raw Shape: {midi_arr}')
        np.save(db_npy_path, midi_arr)
        
    midi_arr_raw = midi_arr
    print(f'[+] Midi Array Shape: {(midi_arr_raw.shape)}')
    print(f'[+] Transforming Midi Input')
    _midi_arr = np.transpose(midi_arr_raw, (0, 2, 1))
    _midi_arr = np.asarray(_midi_arr)
    _midi_arr = np.reshape(_midi_arr, (-1,128))
    print(f'[+] Midi Input: {_midi_arr.shape} ')
    
    return _midi_arr

def fetch_inputs(midi_arr, params):

    x_axis = []
    preds = []

    for i in range (0, midi_arr.shape[0]-params['max_length'], params['time_step']):
        prev = midi_arr[i:i+params['max_length'],...] 
        pred = midi_arr[i+params['max_length'],...] 
        x_axis.append(prev)
        preds.append(pred)

    x_axis = np.asarray(x_axis).astype('float64')
    preds = np.asarray(preds).astype('float64')

    n_samples, max_length, freq = x_axis.shape

    print(f'[i] x_axis shape: {x_axis.shape} ')
    print(f'[i] preds shape: {preds.shape}')

    return n_samples, max_length, freq, x_axis, preds

def sample_midi(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    top = 15
    first = np.random.randint(1,3)

    preds[0:48] = 0
    preds[100:0] = 0

    ind = np.argpartition(preds, -1*top)[-1*top:]
    top_idx_sorted = ind[np.argsort(preds[ind])]

    arr = np.random.uniform(0.0, 0.0, (128))
    arr[top_idx_sorted[0:first]] = 1.0
    arr[top_idx_sorted[first:first+3]] = 0.5

    return arr

def generate_midi(net, midi_arr, start_idx, gen_midi, params, e):
    
    for temp in params['temperature']:
        print(f'----- Temperature: {temp}')
        gen_midi = midi_arr[start_idx : start_idx + params['max_length']]
        for idx in range(680):

            samples = gen_midi[idx:]
            expanded_samples = np.expand_dims(samples, axis=0)

            preds = net.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = sample_midi(preds, temp)
           
            midi_list = []
            midi_list.append(gen_midi)
            midi_list.append(next_array)
            gen_midi = np.vstack(midi_list)

        gen_midi = np.transpose(gen_midi,(1,0))
        output_notes = matrix_to_midi(gen_midi, params, random=1)
        midi_stream = stream.Stream(output_notes)

        midi_fname = params['output_folder'] + f'/composer_output_{e}_{temp}_.mid'
        midi_stream.write('midi', fp=midi_fname)
        parsed = converter.parse(midi_fname)

        for part in parsed.parts:

            part.insert(0, instrument.Piano())
        parsed.write('midi', fp=midi_fname)

    return None



    






        
    















    







