import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import AMC_Read as AMCRead
import LSTM_torch as lstm
from torch.utils.data import DataLoader, TensorDataset
import torch
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

def darken_color(color, factor=0.6):
    """Darkens the given color by multiplying the luminance by the given factor."""
    color_rgb = mcolors.to_rgb(color)
    color_hsv = mcolors.rgb_to_hsv(color_rgb)
    color_hsv[2] *= factor  # Reduce the brightness
    return mcolors.hsv_to_rgb(color_hsv)

# Example usage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modtype=['8PSK','AM-DSB','AM_SSB','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','WBFM']
blue = (0.3059,    0.4745,    0.6549)
darkened_blue = [(1,1,1), darken_color(blue)]
cmapBlue = LinearSegmentedColormap.from_list('my_cmap', darkened_blue)

red = (0.8824,    0.3412,    0.3490)
darkened_red = [(1,1,1), darken_color(red)]
cmapRed = LinearSegmentedColormap.from_list('my_cmap', darkened_red)

def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

def create_folder(folder_path):
    if not os.path.exists(folder_path):

        os.makedirs(folder_path)
        print(f"folder created：{folder_path}")
    else:
        print(f"folder existed：{folder_path}")

def getConfusionMatrixPlot(true_labels, predicted_labels):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    print(cm)

    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=cmapBlue,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(0, width):
        for y in range(0, height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = modtype
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    return plt

filepath_pre = './weights/AMCLSTMweight'
filepath_suff = '0.05'
filepath_suff_read = '0.05-DiRSA'
base = ''
filepath = filepath_pre+filepath_suff+'.h5'

X_train, Y_train, X_test, Y_test, X_val, Y_val, test_SNRs = AMCRead.gendata(seed=12450, filepath_suff=filepath_suff_read, read_origin=False, if_RAF_aug=False)
# X_train, Y_train, X_test, Y_test, X_val, Y_val, test_SNRs = AMCRead.gendata(seed=12450, filepath_suff=filepath_suff_read, read_origin=False, if_RAF_aug=True)
# X_train, Y_train, X_test, Y_test, X_val, Y_val, test_SNRs = AMCRead.gendata(seed=12450, filepath_suff=filepath_suff_read, read_origin=True, if_RAF_aug=False)
# X_train, Y_train, X_test, Y_test, X_val, Y_val, test_SNRs = AMCRead.gendata(seed=12450, filepath_suff=filepath_suff_read, read_origin=True, if_RAF_aug=True)

data_len = 128
batch_size = 400  # training batch size

# Build framework (model)

model = lstm.LSTMCNN()

test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model.load_state_dict(torch.load(filepath))

classes = modtype
snrs=[-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]

folder_path = "./images/LSTM" + filepath_suff
create_folder(folder_path)

acc=[]
for i in range(0, len(snrs)):
    snr = snrs[i]
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]

    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

    # estimate classes
    test_Y_i_hat = model(torch.from_numpy(test_X_i)).detach().numpy()
    width = 4.1
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(folder_path+"/confmat_"+str(snr)+".pdf")
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc.append(1.0*cor/(cor+ncor))
print(acc)

file = open(folder_path+'/overall_acc.txt', "w")
for i in range(0, len(snrs)):
    file.write(str(acc[i]) + '\n')
file.close()