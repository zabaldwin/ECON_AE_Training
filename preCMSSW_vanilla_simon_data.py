import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from qkeras import QActivation,QConv2D,QDense,quantized_bits
import qkeras
from qkeras.utils import model_save_quantized_weights
from keras.models import Model
from keras.layers import *
from telescope import *
from utils import *
import inspect
import json
import os
import sys
import graph

import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import matplotlib.pyplot as plt
import mplhep as hep


p = ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    
    ('--mpath', p.STR)
   
    
    
)

    
remap_8x8 = [ 4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
              24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
              59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]


def get_pams():
    jsonpams={}      
    opt_classes = tuple(opt[1] for opt in inspect.getmembers(tf.keras.optimizers,inspect.isclass))
    for k,v in self.pams.items():
        if type(v)==type(np.array([])):
            jsonpams[k] = v.tolist()
        elif  isinstance(v,opt_classes):
            config = {}
            for hp in v.get_config():
                config[hp] = str(v.get_config()[hp])
            jsonpams[k] = config
        elif  type(v)==type(telescopeMSE8x8):
            jsonpams[k] =str(v) 
        else:
            jsonpams[k] = v 
    return jsonpams

def save_models(autoencoder, name, isQK=False):
    
    #fix all this saving shit
    

    json_string = autoencoder.to_json()
    encoder = autoencoder.get_layer("encoder")
    decoder = autoencoder.get_layer("decoder")
    f'./{model_dir}/{name}.json'
    with open(f'./{model_dir}/{name}.json','w') as f:        f.write(autoencoder.to_json())
    with open(f'./{model_dir}/encoder_{name}.json','w') as f:            f.write(encoder.to_json())
    with open(f'./{model_dir}/decoder_{name}.json','w') as f:            f.write(decoder.to_json())

    autoencoder.save_weights(f'./{model_dir}/{name}.hdf5')
    encoder.save_weights(f'./{model_dir}/encoder_{name}.hdf5')
    decoder.save_weights(f'./{model_dir}/decoder_{name}.hdf5')
    if isQK:
        encoder_qWeight = model_save_quantized_weights(encoder)
        with open(f'{model_dir}/encoder_{name}.pkl','wb') as f:
            pickle.dump(encoder_qWeight,f)
        encoder = graph.set_quantized_weights(encoder,f'{model_dir}/encoder_'+name+'.pkl')
    graph.write_frozen_dummy_enc(encoder,'encoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_dummy_enc(encoder,'encoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_dec(decoder,'decoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)

    graph.plot_weights(autoencoder,outdir = model_dir)
    graph.plot_weights(encoder,outdir = model_dir)
    graph.plot_weights(decoder,outdir = model_dir)
    


def load_matching_state_dict(model, state_dict_path):
    state_dict = tf.compat.v1.train.load_checkpoint(state_dict_path)
    model_variables = model.trainable_variables
    filtered_state_dict = {}
    for var in model_variables:
        var_name = var.name.split(':')[0]
        if var_name in state_dict:
            filtered_state_dict[var_name] = state_dict[var_name]
    tf.compat.v1.train.init_from_checkpoint(state_dict_path, filtered_state_dict)

args = p.parse_args()
model_dir = args.mpath + '_CMSSW'
print(model_dir)

if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

for eLinks in [2,3,4,5]:
     
    bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    
    print(f'Processing Model with {eLinks} eLinks')
    model_dir = os.path.join(args.mpath + '_CMSSW', f'model_{eLinks}_eLinks')
    if not os.path.exists(model_dir):
        os.system("mkdir -p " + model_dir)
    bitsPerOutput = bitsPerOutputLink[eLinks]
    nIntegerBits = 1;
    nDecimalBits = bitsPerOutput - nIntegerBits;
    outputSaturationValue = (1 << nIntegerBits) - 1./(1 << nDecimalBits);
    maxBitsPerOutput = 9
    outputMaxIntSize = 1

    if bitsPerOutput > 0:
        outputMaxIntSize = 1 << nDecimalBits

    outputMaxIntSizeGlobal = 1
    if maxBitsPerOutput > 0:
        outputMaxIntSizeGlobal = 1 << (maxBitsPerOutput - nIntegerBits)

    batch = 1

    n_kernels = 8
    n_encoded=16
    conv_weightBits  = 6 
    conv_biasBits  = 6 
    dense_weightBits  = 6 
    dense_biasBits  = 6 
    encodedBits = 9
    CNN_kernel_size = 3
    padding = tf.constant([[0,0],[0, 1], [0, 1], [0, 0]])


    input_enc = Input(batch_shape=(batch,8,8, 1))
    # sum_input quantization is done in the dataloading step for simplicity
    # sum_input = Input(batch_shape=(batch,1))
    # eta = Input(batch_shape =(batch,1))


    # Quantizing input, 8 bit quantization, 1 bit for integer
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'input_quantization')(input_enc)
    x = tf.pad(
        x, padding, mode='CONSTANT', constant_values=0, name=None
    )
    x = QConv2D(n_kernels,
                CNN_kernel_size, 
                strides=2,padding = 'valid', kernel_quantizer=quantized_bits(bits=conv_weightBits,integer=0,keep_negative=1,alpha=1), bias_quantizer=quantized_bits(bits=conv_biasBits,integer=0,keep_negative=1,alpha=1),
                name="conv2d")(x)

    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'act')(x)

    # x = QActivation("quantized_relu(bits=8,integer=1)", name="act")(x)

    x = Flatten()(x)

    x = QDense(n_encoded, 
               kernel_quantizer=quantized_bits(bits=dense_weightBits,integer=0,keep_negative=1,alpha=1),
               bias_quantizer=quantized_bits(bits=dense_biasBits,integer=0,keep_negative=1,alpha=1),
               name="dense")(x)

    # Quantizing latent space, 9 bit quantization, 1 bit for integer
    x = QActivation(qkeras.quantized_bits(bits = 9, integer = 1),name = 'latent_quantization')(x)

    # x = concatenate([x,sum_input,eta],axis=1)

    latent = x
    if bitsPerOutput > 0 and maxBitsPerOutput > 0:
        latent = tf.minimum(tf.math.floor(latent *  outputMaxIntSize) /  outputMaxIntSize, outputSaturationValue)

    input_dec = Input(batch_shape=(batch,16))
    y = Dense(24)(input_dec)
    y = ReLU()(y)
    y = Dense(64)(y)
    y = ReLU()(y)
    y = Dense(128)(y)
    y = ReLU()(y)
    y = Reshape((4, 4, 8))(y)
    y = Conv2DTranspose(1, (3, 3), strides=(2, 2),padding = 'valid')(y)
    y =y[:,0:8,0:8]
    y = ReLU()(y)
    recon = y

    encoder = keras.Model([input_enc], latent, name="encoder")
    decoder = keras.Model([input_dec], recon, name="decoder")

    cae = Model(
        inputs=[input_enc],
        outputs=decoder([encoder([input_enc])]),
        name="cae"
    )

 
    loss = telescopeMSE8x8
    opt = tf.keras.optimizers.Lion(learning_rate = 0.1,weight_decay = 0.00025)
    cae.compile(optimizer=opt, loss=loss)
#     cae.summary()



    eLink_path = os.path.join(args.mpath,f'model_{eLinks}_eLinks','best-epoch.tf')
    cae.load_weights(eLink_path)
    print('loaded model')
    save_models(cae,args.mname,isQK = True)

