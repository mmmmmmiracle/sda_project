import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input, Lambda, Activation, LSTM
from keras.layers import add, multiply, concatenate,dot
from keras.activations import softmax
from keras import backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

class Sequence():
    def __init__(self,input_shape=[24,1],output_shape=[6],depth=[1,1],hidden_dim=[50,50]):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.depth = depth
        self.hidden_dim = hidden_dim
    
#     def self_attention(self,X):
#         dim = np.int16(np.sqrt(self.hidden_dim[0]))
#         Q = Dense(units=self.hidden_dim[0],activation='tanh')(X)
#         K = Dense(units=self.hidden_dim[0],activation='tanh')(X)
#         V = Dense(units=self.hidden_dim[0],activation='tanh')(X)
#         tmp = dot( [Q,K],axes=-1)
#         tmp = Lambda(lambda x: x *(1/dim))(tmp)
#     #             print(tmp.shape)
#         tmp = Activation('softmax')(tmp)
#     #             print(tmp.shape)
#         encoded = dot( [tmp,Lambda(lambda x: backend.reshape(x,(-1,self.hidden_dim[0],self.input_shape[0])))(V)],axes=-1 )
#         return encoded

    def make_model(self):
        x = Input(batch_shape=(None,self.input_shape[0],1))
        encoded = None
        decoded = None
        output = None
        for i in range(self.depth[0]):
            if i == 0:
                encoded = LSTM(units=self.hidden_dim[0],activation='tanh',return_sequences=True)(x)
            else:
                encoded = LSTM(units=self.hidden_dim[0],activation='tanh',return_sequences=True)(encoded)
        for i in range(self.depth[1]):
            if i == 0:
                decoded = LSTM(units=self.hidden_dim[1],activation='tanh',return_sequences=False)(encoded)
            else:
                C = Lambda(lambda x: backend.repeat(x, self.input_shape[0]), output_shape=(self.input_shape[0], self.hidden_dim[1]))(decoded)
                _xC = concatenate([encoded, C])
                decoded = LSTM(units=self.hidden_dim[1],activation='tanh',return_sequences=False)(_xC)
        output   = Dense(units=self.output_shape[0],activation='relu')(decoded)
        model = Model([x,],[output])
        return model

    def make_model_transformer(self):
        x = Input(batch_shape=(None,self.input_shape[0],1))
        encoded = None
        decoded = None
        output = None
        dim = np.int16(np.sqrt(self.hidden_dim[0]))
        for i in range(self.depth[0]):
            if i == 0:
                encoded = LSTM(units=self.hidden_dim[0],activation='tanh',return_sequences=True)(x)
#                 encoded = Lambda(self.self_attention)(encoded)
                Q = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                K = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                V = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                tmp = dot( [Q,K],axes=-1)
                tmp = Lambda(lambda x: x *(1/dim))(tmp)
    #             print(tmp.shape)
                tmp = Activation('softmax')(tmp)
    #             print(tmp.shape)
                encoded = dot( [tmp,Lambda(lambda x: backend.reshape(x,(-1,self.hidden_dim[0],self.input_shape[0])))(V)],axes=-1 )
    #             print(encoded.shape)
            else:
                encoded = LSTM(units=self.hidden_dim[0],activation='tanh',return_sequences=True)(encoded)
#                 encoded = Lambda(self.self_attention)(encoded)
                Q = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                K = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                V = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                tmp = dot( [Q,K],axes=-1)
                tmp = Lambda(lambda x: x *(1/dim))(tmp)
    #             print(tmp.shape)
                tmp = Activation('softmax')(tmp)
    #             print(tmp.shape)
                encoded = dot( [tmp,Lambda(lambda x: backend.reshape(x,(-1,self.hidden_dim[0],self.input_shape[0])))(V)],axes=-1 )
        for i in range(self.depth[1]):
            if i == 0:
                decoded = LSTM(units=self.hidden_dim[1],activation='tanh',return_sequences=False)(encoded)
            else:
                C = Lambda(lambda x: backend.repeat(x, self.input_shape[0]), output_shape=(self.input_shape[0], self.hidden_dim[1]))(decoded)
                _xC = concatenate([encoded, C])
                decoded = LSTM(units=self.hidden_dim[1],activation='tanh',return_sequences=False)(_xC)
        output   = Dense(units=self.output_shape[0],activation='relu')(decoded)
        print(x,output)
        model = Model([x,],[output,])
        return model

    def make_model_multiheads(self,heads=8,dropout=0.33):
            x = Input(batch_shape=(None,self.input_shape[0],1))
            inputs = []
            encoded = None
            decoded = None
            output = None
            dim = np.int16(np.sqrt(self.hidden_dim[0]))
            encoded_heads = None
            for head in range(heads):
                for i in range(self.depth[0]):
                    if i == 0:
                        encoded = LSTM(units=self.hidden_dim[0],activation='tanh',return_sequences=True)(x)
        #                 encoded = Lambda(self.self_attention)(encoded)
                        Q = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                        K = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                        V = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                        tmp = dot( [Q,K],axes=-1)
                        tmp = Lambda(lambda x: x *(1/dim))(tmp)
            #             print(tmp.shape)
                        tmp = Activation('softmax')(tmp)
            #             print(tmp.shape)
                        encoded = dot( [tmp,Lambda(lambda x: backend.reshape(x,(-1,self.hidden_dim[0],self.input_shape[0])))(V)],axes=-1 )
                        encoded = Dropout(dropout)(encoded)
            #             print(encoded.shape)
                    else:
                        encoded = LSTM(units=self.hidden_dim[0],activation='tanh',return_sequences=True)(encoded)
        #                 encoded = Lambda(self.self_attention)(encoded)
                        Q = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                        K = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                        V = Dense(units=self.hidden_dim[0],activation='tanh')(encoded)
                        tmp = dot( [Q,K],axes=-1)
                        tmp = Lambda(lambda x: x *(1/dim))(tmp)
            #             print(tmp.shape)
                        tmp = Activation('softmax')(tmp)
            #             print(tmp.shape)
                        encoded = dot( [tmp,Lambda(lambda x: backend.reshape(x,(-1,self.hidden_dim[0],self.input_shape[0])))(V)],axes=-1 )
                        encoded = Dropout(dropout)(encoded)
                if head == 0:
                    encoded_heads = encoded
                else:
                    encoded_heads = concatenate([encoded_heads,encoded])
            print(encoded_heads.shape)
            encoded = Dense(units=self.hidden_dim[0],activation='tanh')(encoded_heads)
            encoded = Dropout(dropout)(encoded)
            for i in range(self.depth[1]):
                if i == 0:
                    decoded = LSTM(units=self.hidden_dim[1],activation='tanh',return_sequences=False)(encoded)
                    decoded = Dropout(dropout)(decoded)
                else:
                    C = Lambda(lambda x: backend.repeat(x, self.input_shape[0]), output_shape=(self.input_shape[0], self.hidden_dim[1]))(decoded)
                    _xC = concatenate([encoded, C])
                    decoded = LSTM(units=self.hidden_dim[1],activation='tanh',return_sequences=False)(_xC)
                    decoded = Dropout(dropout)(decoded)
            output   = Dense(units=self.output_shape[0],activation='relu')(decoded)
            print(x,output)
            model = Model([x,],[output,])
            return model
    
# seq = Sequence(depth=(4,4))
# # model = seq.make_model_transformer()
# model = seq.make_model_multiheads(heads = 8)

