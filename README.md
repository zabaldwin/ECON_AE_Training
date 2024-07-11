# ECON_AE_Training
Basic training code for training CAE for HGCAL wafer level compression

## Set up
Build the environment:

```shell
conda env create -f environment.yml
```


### Environmental setup -- PSC | Vera
Create a symbolink to the environment directory

```shell
$ ln -s /hildafs/projects/phy230010p/share/ECON_env ECON_env
```

Create a bash script for sourcing the environment called `env_setup.sh`

```shell
#!/bin/bash
# Activate the environment using the symbo-linked directory
source ~/ECON_env/myenv/bin/activate
echo "Environment setup complete"

# Start an interactive shell to keep the GPU allocation running
exec bash --login
```

Include the following into your `.bashrc` so it will execute everytime you login

```shell
# Check if the script is being sourced from an interactive shell
if [[ $- == *i* ]]; then
   # Check if we are not already in an salloc session by looking for SLURM_JOB_ID
   if [[ -z "$SLURM_JOB_ID" && -z "$SALLOC_ACTIVE" ]]; then
     export SALLOC_ACTIVE=1
     # Run salloc and execute env_setup.sh after allocation
     salloc -p TWIG --gpus=1 -t 12:00:00 bash ~/env_setup.sh
     exit
   fi
fi

# Shows allocation to ensure everything is enabled
squeue -u $USER
```

## CAE Description

The Conditional Autoencoder (CAE) is a quantized encoder & unquantized decoder with conditioning in the latent space for known information. For the HGCAL wafer encoding we have the following conditional information: eta, waferu, waferv, wafertype, sumCALQ, layers. We hot-encode wafertype as there are three possible wafertypes. This gives 8 total conditional variables. In combination with the 16D latent space, this means the decoder takes a 24D latent vector to decode the wafer.


Training the model is done through the `train_CAE_simon_data.py` file. To process data for CMSSW, you must use the CMSSW processing file `dev_CMSSW.py`. The processing modifies the model architecture so that CMSSW apply conditioning without producing errors. There is no change of this for model performance, just how we feed it into CMSSW.

## Example training CAE model:
To train the model:

```shell
python train_CAE_simon_data.py --mname AE  --batchsize 4000 --lr 1e-4 --nepochs 1000 --opath Tele_CAE_biased_90 --optim lion --loss tele --biased --b_percent 0.90
```

Update training w/ configurable pileup percentage in dataset:

```shell 
python train_CAE_simon_data.py --mname vanilla_AE  --batchsize 4000 --lr 3e-4 --nepochs 500 --opath search_pileup_10 --optim lion --loss tele --alloc_geom old --model_per_bit --num_files 100 --biased 0.1
```






To process the model:

```shell
python dev_preCMSSW.py --mname AE --mpath Tele_CAE_biased_90
```
