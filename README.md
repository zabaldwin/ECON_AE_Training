# ECON_AE_Training
Basic training code for training CAE for HGCAL wafer level compression


## Set up
Build the environment: 

conda env create -f environment.yml

## CAE Description

The Conditional Autoencoder (CAE) is a quantized encoder & unquantized decoder with conditioning in the latent space for known information. For the HGCAL wafer encoding we have the following conditional information: eta, waferu, waferv, wafertype, sumCALQ, layers. We hot-encode wafertype as there are three possible wafertypes. This gives 8 total conditional variables. In combination with the 16D latent space, this means the decoder takes a 24D latent vector to decode the wafer.


Training the model is done through the train_CAE_simon_data.py file. To process data for CMSSW, you must use the CMSSW processing file dev_CMSSW.py. The processing modifies the model architecture so that CMSSW apply conditioning without producing errors. There is no change of this for model performance, just how we feed it into CMSSW.  
 
## Example training CAE model:
To train the model:

python train_CAE_simon_data.py --mname AE  --batchsize 4000 --lr 1e-4 --nepochs 1000 --opath Tele_CAE_biased_90 --optim lion --loss tele --biased --b_percent 0.90


To process the model: 

python dev_preCMSSW.py --mname AE --mpath Tele_CAE_biased_90
