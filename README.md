# Adaptive Sampling of k-space in Magnetic Resonance for Rapid Pathology Prediction



## Installation
1. Clone the repository

2. Install the dependencies
```
conda create -n "asmr" python=3.10
pip install -r requirements.txt
pip install -e .
```


## Dataset Preparation 
The dataset can be downloaded from the [fastMRI website](https://fastmri.med.nyu.edu/). Users need to provide a csv file that includes the pathology information. Once the data is ready, set the `datadir` and `split_csv_file` according to the data path. Tools for processing the data are in `data_modules`.



## Usage I: Simulation Environment
We provide sample running scripts in the `scripts/` directory.



### ASMR's Environment and Reward Model

To enable the use of ASMR's environment, we first need to train a reward model. 
The configuration and hyperparamters for training a reward model is located in the `configs/` directory. Set the `datadir` and `split_csv_file` in the YAML file to your data path. 

Trained the reward model:
```
bash train_reward_model.sh 
```



## Usage II: Reinforcement Learning

Once the reward model is ready, we can use the ASMR environment for reinforcement learning. 
The configuration for the ASMR environment is located in the `rl/cfgs/env/` directory. Set the `reward_model_ckpt` in the YAML file to the reward model checkpoint from the previous step.

### Train ASMR Policy

1. Generate the weighted sampler for training, 
```
bash generate_sampler.sh
```
and set the `train_sampler_filename` in the `rl/cfgs/env/*.yaml`  accordingly.

2. Train the ASMR policy
```
bash train_asmr.sh
```



### Evaluate ASMR Policy
To evaluate the trained policy
```
cd rl
python eval_asmr.py  load_from_snapshot_base_dir=<policy-checkpoint-directory>  eval_range=[<start>,<end>]
```



