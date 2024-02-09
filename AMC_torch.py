import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import AMC_Read as AMCRead
import LSTM_torch as lstm
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
modtype=['8PSK','AM-DSB','AM_SSB','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','WBFM']

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"folder created：{folder_path}")
    else:
        print(f"folder existed：{folder_path}")

def train(
    model,
    nb_epoch,
    lr,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=1,
    filepath="",
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-6)
    p1 = int(0.75 * nb_epoch)
    p2 = int(0.9 * nb_epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    best_valid_loss = 1e10
    loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    for epoch_no in range(nb_epoch):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, (x, labels) in enumerate(it, start=1):
                x = Variable(x.to(device))
                labels = Variable(labels.to(device))

                optimizer.zero_grad()

                outputs = model(x)
                loss = loss_fn(outputs, labels)

                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, (x, labels) in enumerate(it, start=1):
                        x = Variable(x.to(device))
                        labels = Variable(labels.to(device))

                        outputs = model(x)
                        loss = loss_fn(outputs, labels)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
            # writer.add_scalar('val_loss', running_loss / 10, epoch * size + step)

    if filepath != "":
        torch.save(model.state_dict(), filepath)

filepath_pre = './weights/AMCLSTMweight'
filepath_suff = '0.05'
filepath_suff_read = '0.05-DiRSA'
filepath = filepath_pre+filepath_suff+'.h5'

X_train, Y_train, X_test, Y_test, X_val, Y_val, test_SNRs = AMCRead.gendata(seed=12450, filepath_suff=filepath_suff_read, read_origin=False, if_RAF_aug=False)
# X_train, Y_train, X_test, Y_test, X_val, Y_val, test_SNRs = AMCRead.gendata(seed=12450, filepath_suff=filepath_suff_read, read_origin=False, if_RAF_aug=True)
# X_train, Y_train, X_test, Y_test, X_val, Y_val, test_SNRs = AMCRead.gendata(seed=12450, filepath_suff=filepath_suff_read, read_origin=True, if_RAF_aug=False)
# X_train, Y_train, X_test, Y_test, X_val, Y_val, test_SNRs = AMCRead.gendata(seed=12450, filepath_suff=filepath_suff_read, read_origin=True, if_RAF_aug=True)

data_len=128

X=0
Y=0
# Set up some params
nb_epoch = 50     # number of epochs to train on
batch_size = 400  # training batch size

# Build framework (model)

model = lstm.LSTMCNN()

learning_rate = 0.001

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

train(
    model,
    nb_epoch,
    learning_rate,
    train_dataloader,
    valid_loader=val_dataloader,
    filepath=filepath,
)