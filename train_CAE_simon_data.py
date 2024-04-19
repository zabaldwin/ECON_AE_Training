import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Layer
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

def filter_for_flat_distribution(dataset, index_i):
    """
    Filters the given TensorFlow dataset to achieve a flat distribution over the specified index i
    of the second element (assumed to be an 8-dimensional tensor) in each dataset element.

    Args:
    - dataset (tf.data.Dataset): The input dataset.
    - index_i (int): The index of the 8-dimensional tensor to achieve a flat distribution over.

    Returns:
    - tf.data.Dataset: A new dataset filtered to achieve a flat distribution across non-zero bins for index_i.
    """
    # Extract the values at index_i from the dataset
    values_to_balance = np.array(list(dataset.map(lambda features, labels: labels[index_i]).as_numpy_iterator()))
    
    # Compute histogram over these values
    counts, bins = np.histogram(values_to_balance, bins=10)
    
    # Identify non-zero bins and determine the minimum count across them for a flat distribution
    non_zero_bins = counts > 0
    min_count_in_non_zero_bins = np.min(counts[non_zero_bins])
    
    # Determine which indices to include for a flat distribution
    indices_to_include = []
    current_counts = np.zeros_like(counts)
    for i, value in enumerate(values_to_balance):
        bin_index = np.digitize(value, bins) - 1
        bin_index = min(bin_index, len(current_counts) - 1)  # Ensure bin_index is within bounds
        if current_counts[bin_index] < min_count_in_non_zero_bins:
            indices_to_include.append(i)
            current_counts[bin_index] += 1
            
    # Convert list of indices to a TensorFlow constant for filtering
    indices_to_include_tf = tf.constant(indices_to_include, dtype=tf.int64)
    
    # Filtering function to apply with the dataset's enumerate method
    def filter_func(index, data):
        return tf.reduce_any(tf.equal(indices_to_include_tf, index))
    
    # Apply filtering to achieve the flat distribution
    filtered_dataset = dataset.enumerate().filter(filter_func).map(lambda idx, data: data)
    
    return filtered_dataset

p = ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    ('--loss', p.STR), ('--nepochs', p.INT),
    ('--opath', p.STR),
    ('--mpath', p.STR),('--prepath', p.STR),('--continue_training', p.STORE_TRUE), ('--batchsize', p.INT),
    ('--lr', {'type': float}),
    ('--num_files', p.INT),('--optim', p.STR),('--eLinks', p.INT),('--emd_pth', p.STR),('--sim_e_cut', p.STORE_TRUE),('--e_cut', p.STORE_TRUE),('--biased', p.STORE_TRUE),('--b_percent', {'type': float}),('--flat_eta', p.STORE_TRUE),('--flat_sum_CALQ', p.STORE_TRUE)
    
    
    
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
    graph.write_frozen_graph_enc(encoder,'encoder_'+name+'.pb',logdir = model_dir)
    graph.write_frozen_graph_enc(encoder,'encoder_'+name+'.pb.ascii',logdir = model_dir,asText=True)
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

def resample_indices(indices, energy, bin_edges, target_count, bin_index):
    bin_indices = indices[(energy > bin_edges[bin_index]) & (energy <= bin_edges[bin_index+1])]
    if len(bin_indices) > target_count:
        return np.random.choice(bin_indices, size=target_count, replace=False)
    else:
        return np.random.choice(bin_indices, size=target_count, replace=True)

def load_data(nfiles,batchsize,eLinks = -1, normalize = True):
    from files import get_rootfiles
    from coffea.nanoevents import NanoEventsFactory
    import awkward as ak
    import numpy as np
    ecr = np.vectorize(encode)
    data_list = []

    # Paths to Simon's dataset
    hostid = 'cmseos.fnal.gov'
    basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
    tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'

    files = get_rootfiles(hostid, basepath)[0:nfiles]


    #loop over all the files
    for i,file in enumerate(files):
        x = NanoEventsFactory.from_root(file, treepath=tree).events()

        min_pt = 10  # replace with your minimum value
        max_pt = 1000  # replace with your maximum value
        gen_pt = ak.to_pandas(x.gen.pt).groupby(level=0).mean()
        mask = (gen_pt['values'] >= min_pt) & (gen_pt['values'] <= max_pt)
        layers = ak.to_pandas(x.wafer.layer)
        layers = layers.loc[layers.index.get_level_values('entry').isin(mask)]

        eta = ak.to_pandas(x.wafer.eta)
        eta = eta.loc[eta.index.get_level_values('entry').isin(mask)]

        waferv = ak.to_pandas(x.wafer.waferv)
        waferv = waferv.loc[waferv.index.get_level_values('entry').isin(mask)]

        waferu = ak.to_pandas(x.wafer.waferu)
        waferu = waferu.loc[waferu.index.get_level_values('entry').isin(mask)]

        wafertype = ak.to_pandas(x.wafer.wafertype)
        wafertype = wafertype.loc[wafertype.index.get_level_values('entry').isin(mask)]

        sumCALQ = ak.to_pandas(x.wafer['CALQ0'])
        sumCALQ = sumCALQ.loc[sumCALQ.index.get_level_values('entry').isin(mask)]


        wafer_sim_energy = ak.to_pandas(x.wafer.simenergy)
        wafer_sim_energy = wafer_sim_energy.loc[wafer_sim_energy.index.get_level_values('entry').isin(mask)]

        wafer_energy = ak.to_pandas(x.wafer.energy)
        wafer_energy = wafer_energy.loc[wafer_energy.index.get_level_values('entry').isin(mask)]



        layers = np.squeeze(layers.to_numpy())
        eta = np.squeeze(eta.to_numpy())/3.1
        waferv = np.squeeze(waferv.to_numpy())/12
        waferu = np.squeeze(waferu.to_numpy())/12
        temp = np.squeeze(wafertype.to_numpy())
        wafertype = np.zeros((temp.size, temp.max() + 1))
        wafertype[np.arange(temp.size), temp] = 1
        sumCALQ = np.squeeze(sumCALQ.to_numpy())
        wafer_sim_energy = np.squeeze(wafer_sim_energy.to_numpy())
        wafer_energy = np.squeeze(wafer_energy.to_numpy())



        for i in range(1,64):
            cur = ak.to_pandas(x.wafer[f'CALQ{int(i)}'])
            cur = cur.loc[cur.index.get_level_values('entry').isin(mask)]
            cur = np.squeeze(cur.to_numpy())
            sumCALQ = sumCALQ + cur

        sumCALQ = np.log(sumCALQ+1)

        inputs = []
        for i in range(64):
            cur = ak.to_pandas(x.wafer['AEin%d'%i])
            cur = cur.loc[cur.index.get_level_values('entry').isin(mask)]
            cur = np.squeeze(cur.to_numpy())
            inputs.append(cur) 
        inputs = np.stack(inputs, axis=-1) #stack all 64 inputs
        inputs = np.reshape(inputs, (-1, 8, 8))
        select_eLinks = {5 : (layers<=11) & (layers>=5) ,
                 4 : (layers==7) | (layers==11),
                 3 : (layers==13),
                 2 : (layers<7) | (layers>13),
                 -1 : (layers>0)}
        inputs = inputs[select_eLinks[eLinks]]
        l =(layers[select_eLinks[eLinks]]-1)/(47-1)
        eta = eta[select_eLinks[eLinks]]
        waferv = waferv[select_eLinks[eLinks]]
        waferu = waferu[select_eLinks[eLinks]]
        wafertype = wafertype[select_eLinks[eLinks]]
        sumCALQ = sumCALQ[select_eLinks[eLinks]]
        wafer_sim_energy = wafer_sim_energy[select_eLinks[eLinks]]
        wafer_energy = wafer_energy[select_eLinks[eLinks]]


        if args.sim_e_cut:
            mask = (wafer_sim_energy[select_eLinks[eLinks]] > 0) 
            inputs = inputs[mask]
            l =l[mask]
            eta = eta[mask]
            waferv = waferv[mask]
            waferu = waferu[mask]
            wafertype = wafertype[mask]
            sumCALQ = sumCALQ[mask]


        elif args.e_cut:
            mask =  (wafer_energy[select_eLinks[eLinks]] > 10)

            inputs = inputs[mask]
            l =l[mask]
            eta = eta[mask]
            waferv = waferv[mask]
            waferu = waferu[mask]
            wafertype = wafertype[mask]
            sumCALQ = sumCALQ[mask]

        elif args.biased:
            # New: Resampling based
            filtered_energy = wafer_sim_energy[wafer_sim_energy > 1]

            # Number of samples for demonstration
            num_samples = len(filtered_energy)  # Ensure this matches with your actual data

            # Create a list of indices representing each sample
            indices = np.arange(num_samples)

            # Define bin edges based on energy
            bin_edges = np.histogram_bin_edges(filtered_energy, bins=3)

            # Function to resample indices for each bin
            

            # Target sample count for each bin
            target_sample_count = 1000

            # Resample indices for each bin
            resampled_indices = np.array([], dtype=int)

            for i in range(len(bin_edges)-1):
                resampled_bin_indices = resample_indices(indices, filtered_energy, bin_edges, target_sample_count, i)
                resampled_indices = np.concatenate([resampled_indices, resampled_bin_indices])
            n_pileup = int(3000 * (1/args.b_percent - 1))
            resampled_indices = np.concatenate([resampled_indices,np.random.choice(np.where(wafer_sim_energy < 1)[0],n_pileup)])

            # Now use resampled_indices to select samples from each feature array
            inputs = inputs[resampled_indices]
            l = layers[resampled_indices]
            eta = eta[resampled_indices]
            waferv = waferv[resampled_indices]
            waferu = waferu[resampled_indices]
            wafertype = wafertype[resampled_indices]
            sumCALQ = sumCALQ[resampled_indices]
            
            # Old: Cut based
#             mask = (wafer_sim_energy > 0) 
#             indices_passing = np.where(mask)[0]
#             indices_not_passing = np.where(~mask)[0]
            
#             if args.b_percent is not None:
#                 k = args.b_percent /(1-args.b_percent)
#             else: 
#                 k = 3
#             desired_not_passing_count = int(len(indices_passing) / k) 
#             selected_not_passing_indices = np.random.choice(indices_not_passing, size=desired_not_passing_count, replace=False)

#             new_mask_indices = np.concatenate((indices_passing, selected_not_passing_indices))
#             mask = np.zeros_like(wafer_sim_energy, dtype=bool)
#             mask[new_mask_indices] = True
         

#             inputs = inputs[mask]
#             l =l[mask]
#             eta = eta[mask]
#             waferv = waferv[mask]
#             waferu = waferu[mask]
#             wafertype = wafertype[mask]
#             sumCALQ = sumCALQ[mask]
        data_list.append([inputs,eta,waferv,waferu,wafertype,sumCALQ,l])


    inputs_list = []
    eta_list = []
    waferv_list = []
    waferu_list = []
    wafertype_list = []
    sumCALQ_list = []
    layer_list = []

    for item in data_list:
        inputs, eta, waferv, waferu, wafertype, sumCALQ,layers = item
        inputs_list.append(inputs)
        eta_list.append(eta)
        waferv_list.append(waferv)
        waferu_list.append(waferu)
        wafertype_list.append(wafertype)
        sumCALQ_list.append(sumCALQ)
        layer_list.append(layers)

    concatenated_inputs = np.expand_dims(np.concatenate(inputs_list),axis = -1)
    concatenated_eta = np.expand_dims(np.concatenate(eta_list),axis = -1)
    concatenated_waferv = np.expand_dims(np.concatenate(waferv_list),axis = -1)
    concatenated_waferu = np.expand_dims(np.concatenate(waferu_list),axis = -1)
    concatenated_wafertype = np.concatenate(wafertype_list)
    concatenated_sumCALQ = np.expand_dims(np.concatenate(sumCALQ_list),axis = -1)
    concatenated_layers = np.expand_dims(np.concatenate(layer_list),axis = -1)

    concatenated_cond = np.hstack([concatenated_eta,concatenated_waferv,concatenated_waferu, concatenated_wafertype, concatenated_sumCALQ,concatenated_layers])


    all_dataset = tf.data.Dataset.from_tensor_slices((concatenated_inputs, concatenated_cond)
    ).take(1000000)

    total_size = len(all_dataset)  # Replace with your dataset's total size
    print('total size: ',total_size)
    
    # Define your splitting ratio
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Create the training dataset
    train_dataset = all_dataset.take(train_size)

    # Create the test dataset
    test_dataset = all_dataset.skip(train_size).take(test_size)

    train_loader = train_dataset.batch(batchsize).shuffle(buffer_size=train_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_loader = test_dataset.batch(batchsize).shuffle(buffer_size=test_size).prefetch(buffer_size=tf.data.AUTOTUNE)


    return train_loader, test_loader
    
class keras_pad(Layer):
    def call(self, x):
        padding = tf.constant([[0,0],[0, 1], [0, 1], [0, 0]])
        return tf.pad(
        x, padding, mode='CONSTANT', constant_values=0, name=None
    )
    
    
class keras_minimum(Layer):
    def call(self, x, sat_val = 1):
        return tf.minimum(x,sat_val)
    
    
class keras_floor(Layer):
    def call(self, x):
        if isinstance(x, tf.SparseTensor):
            x = tf.sparse.to_dense(x)
            
        return tf.math.floor(x)
      
    
    
    
args = p.parse_args()
model_dir = args.opath
if not os.path.exists(model_dir):
    os.system("mkdir -p "+model_dir)

# Loop through each number of eLinks
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
    


    input_enc = Input(batch_shape=(batch,8,8, 1), name = 'Wafer')
    # sum_input quantization is done in the dataloading step for simplicity
    
    cond = Input(batch_shape=(batch, 8), name = 'Cond')


    # Quantizing input, 8 bit quantization, 1 bit for integer
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'input_quantization')(input_enc)
    x = keras_pad()(x)
#     x = tf.pad(
#         x, padding, mode='CONSTANT', constant_values=0, name=None
#     )
    x = QConv2D(n_kernels,
                CNN_kernel_size, 
                strides=2,padding = 'valid', kernel_quantizer=quantized_bits(bits=conv_weightBits,integer=0,keep_negative=1,alpha=1), bias_quantizer=quantized_bits(bits=conv_biasBits,integer=0,keep_negative=1,alpha=1),
                name="conv2d")(x)
    
    x = QActivation(quantized_bits(bits = 8, integer = 1),name = 'act')(x)
    

    x = Flatten()(x)

    x = QDense(n_encoded, 
               kernel_quantizer=quantized_bits(bits=dense_weightBits,integer=0,keep_negative=1,alpha=1),
               bias_quantizer=quantized_bits(bits=dense_biasBits,integer=0,keep_negative=1,alpha=1),
               name="dense")(x)

    # Quantizing latent space, 9 bit quantization, 1 bit for integer
    x = QActivation(qkeras.quantized_bits(bits = 9, integer = 1),name = 'latent_quantization')(x)
    latent = x
    if bitsPerOutput > 0 and maxBitsPerOutput > 0:
        latent = keras_floor()(latent *  outputMaxIntSize)
        latent = keras_minimum()(latent/outputMaxIntSize, sat_val = outputSaturationValue)
#         latent = tf.minimum(tf.math.floor(latent *  outputMaxIntSize) /  outputMaxIntSize, outputSaturationValue)

    latent = concatenate([latent,cond],axis=1)
   

    
    input_dec = Input(batch_shape=(batch,24))
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
    
    encoder = keras.Model([input_enc,cond], latent, name="encoder")
    decoder = keras.Model([input_dec], recon, name="decoder")

    cae = Model(
        inputs=[input_enc,cond],
        outputs=decoder([encoder([input_enc,cond])]),
        name="cae"
    )

    if args.loss == 'mse':
        loss=mean_mse_loss
    elif args.loss == 'tele':
        print('Using tele')
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
        print('loaded model')





    print('Loading Data')
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
       
        total_loss_train = 0

        for wafers, cond in train_loader:

            loss = cae.train_on_batch([wafers,cond], wafers)
            total_loss_train = total_loss_train + loss

        total_loss_val = 0 
        for wafers, cond in test_loader:


            loss = cae.test_on_batch([wafers, cond], wafers)

            total_loss_val = total_loss_val+loss


        total_loss_train = total_loss_train#/(len(train_loader))
        total_loss_val = total_loss_val#/(len(test_loader))
        if epoch % 25 == 0:
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

        if total_loss_val < best_val_loss:
            if epoch % 25 == 0:
                print('New Best Model')
            best_val_loss = total_loss_val
            cae.save_weights(os.path.join(model_dir, 'best-epoch.tf'.format(epoch)))
            encoder.save_weights(os.path.join(model_dir, 'best-encoder-epoch.tf'.format(epoch)))
            decoder.save_weights(os.path.join(model_dir, 'best-decoder-epoch.tf'.format(epoch)))
    save_models(cae,args.mname,isQK = True)

