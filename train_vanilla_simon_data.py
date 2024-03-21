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
    ('--loss', p.STR), ('--nepochs', p.INT),
    ('--opath', p.STR),
    ('--mpath', p.STR),('--prepath', p.STR),('--continue_training', p.STORE_TRUE), ('--batchsize', p.INT),
    ('--lr', {'type': float}),
    ('--num_files', p.INT),('--pretrain_model', p.STORE_TRUE),('--optim', p.STR),('--eLinks', p.INT),('--emd_pth', p.STR),('--sim_e_cut', p.STORE_TRUE),('--e_cut', p.STORE_TRUE),('--biased', p.STORE_TRUE),('--b_percent', {'type': float})
    
    
    
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
def mean_mse_loss(y_true, y_pred):
    
    max_values = tf.reduce_max(y_true[:,], axis=1)
    
    y_true = tf.gather(K.reshape(y_true,(-1,64)),remap_8x8,axis=-1)
    y_pred = tf.gather(K.reshape(y_pred,(-1,64)),remap_8x8,axis=-1)
    # Calculate the squared difference between predicted and target values
    squared_diff = tf.square(y_pred - y_true)

    # Calculate the MSE per row (reduce_mean along axis=1)
    mse_per_row = tf.reduce_mean(squared_diff, axis=1)
    weighted_mse_per_row = mse_per_row * max_values
    

    # Take the mean of the MSE values to get the overall MSE loss
    mean_mse_loss = tf.reduce_mean(weighted_mse_per_row)
    return mean_mse_loss

def get_emd_loss(mpath):
    
    
    emd_model = tf.keras.models.load_model(mpath)
    emd_model.trainable = False
                 
    
  
    def emd_loss(y_true, y_pred):
        return tf.math.abs(emd_model([y_true, y_pred]))
  
    return emd_loss


def load_data(nfiles,batchsize,eLinks = -1, normalize = True):
    from files import get_rootfiles
    from coffea.nanoevents import NanoEventsFactory
    import awkward as ak
    import numpy as np
    ecr = np.vectorize(encode)
    data_list = []
    

    hostid = 'cmseos.fnal.gov'
    basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
    tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'

    files = get_rootfiles(hostid, basepath)[0:nfiles]
    
    #loop over all the files
    for i,file in enumerate(files):
        #open the file
        x = NanoEventsFactory.from_root(file, treepath=tree).events()
        layers = np.squeeze(ak.to_pandas(x.wafer.layer).to_numpy())
        inputs = []
        for i in range(64):
            inputs.append(ak.to_numpy(ak.flatten(x.wafer['AEin%d'%i]))) 

        inputs = np.stack(inputs, axis=-1) #stack all 64 inputs
        inputs = np.reshape(inputs, (-1, 8, 8))
        select_eLinks = {5 : (layers<=11) & (layers>=5) ,
                 4 : (layers==7) | (layers==11),
                 3 : (layers==13),
                 2 : (layers<7) | (layers>13),
                 -1 : (layers>0)}
        inputs = inputs[select_eLinks[eLinks]]
        
        if args.sim_e_cut:
            wafer_sim_energy = ak.to_numpy(ak.flatten(x.wafer.simenergy))
            mask = (wafer_sim_energy[select_eLinks[eLinks]] > 0) 
            inputs = inputs[mask]
            

        elif args.e_cut:
            wafer_energy = ak.to_numpy(ak.flatten(x.wafer.energy))
            mask =  (wafer_energy[select_eLinks[eLinks]] > 10)
            
            inputs = inputs[mask]

        elif args.biased:
            wafer_sim_energy = ak.to_numpy(ak.flatten(x.wafer.simenergy))[select_eLinks[eLinks]]
            mask = (wafer_sim_energy > 0) 
            indices_passing = np.where(mask)[0]
            indices_not_passing = np.where(~mask)[0]
            print(len(indices_passing) )
            if args.b_percent is not None:
                k = args.b_percent /(1-args.b_percent)
            else: 
                k = 3
            desired_not_passing_count = int(len(indices_passing) / k) 
            print(desired_not_passing_count)
            selected_not_passing_indices = np.random.choice(indices_not_passing, size=desired_not_passing_count, replace=False)

            new_mask_indices = np.concatenate((indices_passing, selected_not_passing_indices))
            mask = np.zeros_like(wafer_sim_energy, dtype=bool)
            mask[new_mask_indices] = True
            inputs = inputs[mask]
            
           
        data_list.append(inputs)

    data_tensor = tf.convert_to_tensor(np.concatenate(data_list), dtype=tf.float32)
    data_tensor = data_tensor[0:int(1000000)]
    print(len(data_tensor))
    train_size = int(0.9 * len(data_tensor))
    test_size = len(data_tensor) - train_size

    # Split the data into training and test sets
    train_data, test_data = tf.split(data_tensor, [train_size, test_size], axis=0)

    

    # Create data loaders for training and test data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
    train_loader = train_dataset.batch(batchsize).shuffle(buffer_size=train_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data))
    test_loader = test_dataset.batch(batchsize).shuffle(buffer_size=test_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_loader, test_loader




args = p.parse_args()
model_dir = args.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

for eLinks in [2,3,4,5]:
     
    bitsPerOutputLink = [0, 1, 3, 5, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    
    print(f'Training Model with {eLinks} eLinks')
    model_dir = os.path.join(args.opath, f'model_{eLinks}_eLinks')
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

    batch = args.batchsize

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
#     y = Dense(64)(y)
#     y = ReLU()(y)
#     y = Dense(64)(y)
#     y = ReLU()(y)
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

    if args.loss == 'mse':
        loss=mean_mse_loss
    elif args.loss == 'tele':
        loss = telescopeMSE8x8
    elif args.loss == 'emd':
        loss = get_emd_loss(args.emd_pth)

    print(args.optim)
    if args.optim == 'adam':
        print('Using ADAM Optimizer')
        opt = tf.keras.optimizers.Adam(learning_rate = args.lr,weight_decay = 0.000025)
    elif args.optim == 'lion':
        print('Using Lion Optimizer')
        opt = tf.keras.optimizers.Lion(learning_rate = args.lr,weight_decay = 0.00025)

    cae.compile(optimizer=opt, loss=loss)
    cae.summary()



    # Loading Model
    if args.continue_training:

        cae.load_weights(args.mpath)
        start_epoch = int(args.mpath.split("/")[-1].split(".")[-2].split("-")[-1]) + 1
        print(f"Continuing training from epoch {start_epoch}...")
    elif args.mpath:
        cae.load_weights(args.mpath)
    #     load_matching_state_dict(cae, args.mpath)
        print('loaded model')





    print('Loading Data')
    if args.batchsize != 1:
        train_loader, test_loader = load_data(args.num_files,batch,eLinks =eLinks)
    print('Data Loaded')


    best_val_loss = 1e9

    all_train_loss = []
    all_val_loss = []


    if args.continue_training:
        cut_path = args.mpath.rsplit('/', 2)[0] + '/'
        loss_dict = {'train_loss': pd.read_csv(os.path.join(cut_path,'loss.csv'))['train_loss'].tolist(), 
     'val_loss': pd.read_csv(os.path.join(cut_path,'loss.csv'))['val_loss'].tolist()}
    else:
        start_epoch = 1
        loss_dict = {'train_loss': [], 'val_loss': []}





    for epoch in range(start_epoch, args.nepochs):
        if epoch == 20:
            if args.pretrain_model:
                print('Beginnning Fine Tuning')
                if args.optim == 'adam':
                    opt = tf.keras.optimizers.Adam(learning_rate = args.lr,weight_decay = 0.000025)
                elif args.optim == 'lion':
                    opt = tf.keras.optimizers.Lion(learning_rate = args.lr,weight_decay = 0.00025)
                cae.compile(optimizer=opt, loss=telescopeMSE8x8)

                cae.load_weights(model_dir+'/best-epoch.tf')
                print('Loaded Best Pretrained Model')



        total_loss_train = 0

        for wafers in train_loader:

            loss = cae.train_on_batch([wafers], wafers)
            total_loss_train = total_loss_train + loss

        total_loss_val = 0 
        for wafers in test_loader:


            loss = cae.test_on_batch([wafers], wafers)

            total_loss_val = total_loss_val+loss


        total_loss_train = total_loss_train/(len(train_loader))
        total_loss_val = total_loss_val/(len(test_loader))
        print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
            epoch, total_loss_train,  total_loss_val))



        loss_dict['train_loss'].append(total_loss_train)
        loss_dict['val_loss'].append(total_loss_val)

        df = pd.DataFrame.from_dict(loss_dict)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['train_loss'], label='Training Loss')
        plt.plot(df['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Saving the plot in the same directory as the loss CSV
        plot_path = f"{model_dir}/loss_plot.png"
        plt.savefig(plot_path)


        df.to_csv("%s/" % model_dir + "/loss.csv")

#         cae.save_weights(os.path.join(model_dir, f'epoch-{epoch}.tf'))
        if total_loss_val < best_val_loss:
            print('New Best Model')
            best_val_loss = total_loss_val
            cae.save_weights(os.path.join(model_dir, 'best-epoch.tf'.format(epoch)))

    # tf.saved_model.save(encoder, os.path.join(model_dir, 'best-encoder'))
    # tf.saved_model.save(decoder, os.path.join(model_dir, 'best-decoder'))
    save_models(cae,args.mname,isQK = True)

