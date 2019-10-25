# import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, LSTM, Lambda, RepeatVector, Reshape
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import objectives
import glob

############################ create model #####################################


def get_notes():
    """
    Get all the notes and chords from
    the midi files in the ./midi_songs directory
    """
    notes = []

    for file in glob.glob("/content/midis/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('/content/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(
        network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def lstm_vae(input_dim,
             timesteps,
             batch_size,
             latent_dim,
             n_vocab,
             epsilon_std=1.):

    input_layer = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(512, return_sequences=True)(input_layer)
    h = LSTM(256)(h)
    h = Dense(128, activation='relu')(h)
    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        batch = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch, latent_dim))
        return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

    z = Lambda(sampling)([z_mean, z_log_sigma])
    # decoded layer
    h_decoded = RepeatVector(timesteps)(z)

    h_decoded = LSTM(256, return_sequences=True)(h_decoded)
    h_decoded = LSTM(512, return_sequences=True)(h_decoded)
    h_decoded = LSTM(128)(h_decoded)
    h_decoded_mean = Dense(n_vocab, activation='sigmoid')(h_decoded)

    # end-to-end autoencoder
    vae = Model(input_layer, h_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma -
                                 K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        print(loss.shape)
        return loss

    vae.compile(optimizer='adam', loss=vae_loss)

    return vae


def get_data():
    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    return network_input, network_output, n_vocab


x, y, n_vocab = get_data()

input_dim = x.shape[-1]
timesteps = x.shape[1]


vae = lstm_vae(input_dim,
               timesteps=timesteps,
               batch_size=128,
               latent_dim=128,
               n_vocab=n_vocab,
               epsilon_std=1.)

plot_model(vae, to_file='/content/vae_mlp_m.png', show_shapes=True)
vae.fit(x, y, epochs=20)


########################## create generator ###################################
def generate(vae):
    """ Generate a piano midi file """
    # load the notes used to train the model
    with open('/content/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences2(
        notes, pitchnames, n_vocab)

    prediction_output = generate_notes2(
        vae, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)


def prepare_sequences2(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(
        network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


def generate_notes2(vae, network_input, pitchnames, n_vocab):
    """
    Generate notes from the neural network based on a sequence of notes
    """
    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note)
                       for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = vae.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output):
    """
    convert the output from the prediction to notes and create a midi file
    from the notes
    """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='/content/test_output.mid')


generate(vae)
