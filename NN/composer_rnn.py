
import os

import keras
from keras import layers
from keras import models
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import *

import tensorflow as tf 


class RNN_:

    def __init__(self, params):

        self.input_nodes = int(params['input_nodes'])
        self.hidden_nodes_1 = int(self.input_nodes // 2)
        self.hidden_nodes_2 = int(self.input_nodes // 4)
        self.hidden_nodes_out = int(self.input_nodes // 8)

        self.opt = keras.optimizers.RMSprop(lr=0.007)

        self.max_len = params['max_length']
        self.piano_roll = params['piano_roll']

        self.midi_shape = (self.max_len, self.piano_roll)
        self.input_midi = keras.Input(self.midi_shape)


    def construct_network(self):

        # initial LSTM
        x = layers.LSTM(self.input_nodes, return_sequences=True, unit_forget_bias=True)(self.input_midi)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization() (x)
        x = layers.Dropout(0.3)(x)

        # ----------------------------------------- #

        # secondary LSTM
        x = layers.LSTM(self.input_nodes, return_sequences=True, unit_forget_bias=True)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization() (x)
        x = layers.Dropout(0.6)(x)

        # attention
        attn = layers.Dense(1, activation='tanh')(x)
        attn = layers.Flatten()(attn)
        attn = layers.Activation('softmax')(attn)
        attn = layers.RepeatVector(self.input_nodes)(attn)
        attn = layers.Permute([2, 1])(attn)

        # attention context
        context = layers.Multiply()([x, attn])
        x = layers.Dense((self.input_nodes))(context)

        # ---------------------------------------- #

        # third LSTM
        x = layers.LSTM(self.input_nodes, return_sequences=True, unit_forget_bias=True)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization() (x)
        x = layers.Dropout(0.9)(x)

        # attention
        attn = layers.Dense(1, activation='tanh')(x)
        attn = layers.Flatten()(attn)
        attn = layers.Activation('softmax')(attn)
        attn = layers.RepeatVector(self.input_nodes)(attn)
        attn = layers.Permute([2, 1])(attn)

        # attention context
        context = layers.Multiply()([x, attn])
        x = layers.Dense((self.hidden_nodes_1))(context)

        # fully connected layers
        x = layers.Dense(self.hidden_nodes_1)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization() (x)
        x = layers.Dropout(0.22)(x)

        # ------------------------------------------- #

        # fourth LSTM
        x = layers.LSTM(self.hidden_nodes_1, return_sequences=True, unit_forget_bias=True)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization() (x)
        x = layers.Dropout(0.22)(x)

        # fully connected layers
        x = layers.Dense(self.hidden_nodes_1)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization() (x)
        x = layers.Dropout(0.22)(x)

        # attention
        attn = layers.Dense(1, activation='tanh')(x)
        attn = layers.Flatten()(attn)
        attn = layers.Activation('softmax')(attn)
        attn = layers.RepeatVector(self.hidden_nodes_1)(attn)
        attn = layers.Permute([2, 1])(attn)

        # attention context
        context = layers.Multiply()([x, attn])
        x = layers.Dense(self.hidden_nodes_2)(context)    

        # fully connected layers
        x = layers.Dense(self.hidden_nodes_2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.22)(x)

        # fully connected layers
        x = layers.Dense(self.hidden_nodes_2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.22)(x)

        # ------------------------------------------- #

        # fifth LSTM
        x = layers.LSTM(self.hidden_nodes_out, unit_forget_bias=True)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization() (x)
        x = layers.Dropout(0.22)(x)

        # fully connected out
        x = layers.Dense(self.piano_roll, activation='softmax')(x) 


        # instantiate and compile model
        model = Model(self.input_midi, x)
        model.summary()
        model.compile(loss='categorical_crossentropy', 
                      optimizer=self.opt,
                      metrics=['acc'])

        return model



    














