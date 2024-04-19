# DiRSA
Diffusion-based Radio Signal Augmentation (DiRSA) algorithm is proposed to significantly expand the training dataset, especially small datasets, while preserving
the essential features of the radio signals. We will briefly introduce our code files in order of use. The RadioML2016.10a dataset download: https://www.deepsig.ai/datasets/ 
## General process
First, set the variable "iftrain" to True and run "DiRSA_exe.py" to train Diffusion Models of each modulation. Then, set the "make_aug_dataset" to True and run "DiRSA_exe.py" again to make augmentation datasets. Finally, run "AMC_torch.py" to train the AMC model by augmentation datasets, and run "AMC_figure.py" to evaluate the AMC model. Please download and place the ""RadioML2016.10a_dict.pkl"" dataset in the running directory, and pay attention to set the paths of all generated files.  
## DiRSA_exe.py
Train DDPM of DiRSA or using DDPM to make augmentation datasets.  
### config.yaml
Configs of diffusion training.  
### DiRSA_read.py
Read "RadioML2016.10a_dict.pkl" dataset. Divide, save and read training set, test set, validation set according to requirements, and use rotation and flipping for augmentation of training set.  
### DiRSA_model.py
Details of DiRSA's base model.  
### DiRSA_diff_models.py
Details of DiRSA's DDPM.  
### DiRSA_utils.py
Details of DiRSA's training and augmenting.  
### visualize_examples.ipynb
Read augmented datasets, visualize some samples.  
## AMC_torch.py
Train AMC model, based on rotation and flipping, DiRSA or a mixing augmentation method.  
### AMC_Read.py
Read "RadioML2016.10a_dict.pkl" or augmented dataset. Divide, save and read training set, test set, validation set according to requirements. Apply rotation and flipping for augmentation according to requirements.  
### AMC_figure.py
Evaluate accuracies of AMC model and generate confusion matrices.  
### LSTM_torch.py
Details of LSTM network of AMC.  


