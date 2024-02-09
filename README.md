# DiRSA
Diffusion-based Radio Signal Augmentation (DiRSA) algorithm
We will briefly introduce our code files in order of use.
##DiRSA_exe.py
Read RadioML2016.10a_dict.pkl, train DDPM of DiRSA or using DDPM to make augmentation datasets.
###config.yaml
Saving configs of diffusion training.
###DiRSA_read.py
Read datasets, divide training set, test set and validation set according to requirements for diffusion training, and use rotation and flipping for augmentation of training set.
