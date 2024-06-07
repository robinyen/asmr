# This is a script to train the ASMR model on the knee environment.

source ~/.bashrc
conda activate asmr

python ../rl/train_asmr.py  project_name=asmr_example  env=knee_env   model=asmr_networks  





