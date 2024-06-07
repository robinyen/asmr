# This is an example of training a reward model for knee data
source ~/.bashrc
conda activate asmr

CONFIG_PATH=../configs/knee_rm.yaml
python ../kspace_classifier/bin/train_classifiers.py --config_path=${CONFIG_PATH} 




