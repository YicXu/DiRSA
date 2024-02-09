import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle as cPickle
modtype=['8PSK','AM-DSB','AM_SSB','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','WBFM']

def normalize(X):
    std_dev = np.std(X, axis=(1, 2), keepdims=True)

    k = 1.0 / std_dev

    normalized_data = X * k

    return normalized_data

def rotate_data(X_train, rotate_matrix):
    X_train_rotate = []
    for i in range(0, len(X_train)):
        X_train_rotate.append(np.dot(rotate_matrix, X_train[i]))
    return np.array(X_train_rotate, dtype='float32')

def flip_data(X_train, flip_matrix):
    X_train_flip = []
    for i in range(0, len(X_train)):
        X_train_flip.append((np.multiply(X_train[i].transpose(), flip_matrix)).transpose())
    return np.array(X_train_flip, dtype='float32')

def get_dataloader(seed=2023, nfold=None, batch_size=400, missing_ratio=0.1, make_aug_dataset=False, sampling_scale=1, test_scale=0.1, k=1, mix_k=1,
                   filepath_suff='', new_data=False, val_scale=0.1):

    np.random.seed(seed)
    observed_pt = []
    gt_masks = []

    pathTrain_idx = '/home/xyc/CSDI-IQ/data/DiffTrain_idx'+filepath_suff+'.npy'
    pathVal_idx = '/home/xyc/CSDI-IQ/data/DiffVal_idx'+filepath_suff+'.npy'
    pathTest_idx = '/home/xyc/CSDI-IQ/data/DiffTest_idx'+filepath_suff+'.npy'

    f = 0
    Xd = cPickle.load(open('/home/xyc/AMC/RML2016.10a_dict.pkl', 'rb'), encoding="latin-1")
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    print(mods, snrs)
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))

    stack_X = np.vstack(X)
    data_types = len(snrs)*len(mods)
    data_len = len(stack_X)//data_types

    ct = np.arange(0, len(stack_X))
    temp_snrs = list(map(lambda x: int(lbl[x][1]), ct))
    temp_snrs = np.array(temp_snrs)

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    temp_Y = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), ct)))
    temp_Y = np.array(temp_Y)

    if os.path.exists(pathTrain_idx) and os.path.exists(pathTest_idx) and os.path.exists(pathVal_idx) and (not new_data):
        train_indices = np.load(pathTrain_idx)
    else:
        train_idx = np.random.choice(range(0, data_len), size=int(data_len * sampling_scale), replace=False)
        test_idx = np.random.choice(list(set(range(0, data_len)) - set(train_idx)), size=int(data_len * test_scale), replace=False)
        val_idx = np.random.choice(list(set(range(0, data_len)) - set(train_idx) - set(test_idx)), size=int(data_len * val_scale), replace=False)

        train_indices = train_idx
        val_indices = val_idx
        test_indices = test_idx

        for i in range(1, data_types):
            train_idx = np.random.choice(range(0, data_len), size=int(data_len * sampling_scale), replace=False)
            test_idx = np.random.choice(list(set(range(0, data_len)) - set(train_idx)), size=int(data_len * test_scale), replace=False)
            val_idx = np.random.choice(list(set(range(0, data_len)) - set(train_idx) - set(test_idx)), size=int(data_len * val_scale), replace=False)
            train_indices = np.concatenate((train_indices, train_idx + i * data_len), axis=0)
            val_indices = np.concatenate((val_indices, val_idx + i * data_len), axis=0)
            test_indices = np.concatenate((test_indices, test_idx + i * data_len), axis=0)

        np.save(pathTrain_idx, train_indices)
        np.save(pathTest_idx, test_indices)
        np.save(pathVal_idx, val_indices)

    observed_values = stack_X[train_indices]
    test_SNRs = temp_snrs[train_indices]
    Y_test = temp_Y[train_indices]

    observed_values = normalize(observed_values)

    rotate1 = np.array([[0, -1], [1, 0]])
    rotate2 = np.array([[-1, 0], [0, -1]])
    rotate3 = np.array([[0, 1], [-1, 0]])
    x1 = rotate_data(observed_values, rotate1)
    x2 = rotate_data(observed_values, rotate2)
    x3 = rotate_data(observed_values, rotate3)

    # Flip
    flip1 = np.array([-1, 1])
    flip2 = np.array([1, -1])
    flip3 = np.array([-1, -1])
    x4 = flip_data(observed_values, flip1)
    x5 = flip_data(observed_values, flip2)
    x6 = flip_data(observed_values, flip3)

    observed_values = np.concatenate([observed_values, x1, x2, x3, x4, x5, x6], axis=0)
    test_SNRs = np.tile(test_SNRs, 7)
    Y_test = np.tile(Y_test, (7, 1))

    observed_values = np.tile(observed_values, (mix_k, 1, 1))
    test_SNRs = np.tile(test_SNRs, mix_k)
    Y_test = np.tile(Y_test, (mix_k, 1))

    observed_masks = ~np.isnan(observed_values[:, :, :])
    for i in range(0, len(observed_values)):
        selected_intervals = []
        total_length = 0
        n=1
        max_length = int(128 * missing_ratio)

        while total_length < max_length and len(selected_intervals) < n:
            length = max_length//n

            start = np.random.randint(0, 128 - length)

            is_overlap = any(start < end and start + length > end or start <= begin and start + length >= begin for begin, end in selected_intervals)

            if not is_overlap:
                selected_intervals.append((start, start + length))
                total_length += length

        masks = ~np.isnan(observed_values[i, 0, :])

        for start, end in selected_intervals:
            masks[start:end] = 0

        masks = np.vstack((masks, masks))
        gt_masks.append(masks)

    gt_masks = np.array(gt_masks)
    gt_masks = gt_masks.astype("float32")
    observed_masks = observed_masks.astype("float32")

    for i in range(0, len(observed_values)):
        pt = np.arange(128)
        observed_pt.append(pt)

    observed_pt = np.array(observed_pt)

    train_num = 0.9
    val_num = 0.1
    if (make_aug_dataset):
        test_dataloaders=[]
        Y_test_labels = np.array(torch.argmax(torch.tensor(Y_test), dim=1))
        for i in range(0, len(mods)):
            test_idx = np.where(Y_test_labels==i)
            test_observed_values = observed_values[test_idx]
            test_observed_masks = observed_masks[test_idx]
            test_gt_masks = gt_masks[test_idx]
            test_observed_pt = observed_pt[test_idx]
            Y_test_t = Y_test[test_idx]
            test_SNRs_t = test_SNRs[test_idx]
            test_dataset = TensorDataset(torch.from_numpy(test_observed_values), torch.from_numpy(test_observed_masks),
                                         torch.from_numpy(test_gt_masks),
                                         torch.from_numpy(test_observed_pt), torch.from_numpy(Y_test_t),
                                         torch.from_numpy(test_SNRs_t))
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            test_dataloaders.append(test_dataloader)

        return test_dataloaders

    else:
        Y_test_labels = np.array(torch.argmax(torch.tensor(Y_test), dim=1))
        train_dataloaders=[]
        val_dataloaders=[]
        for i in range(0, len(mods)):
            idx = np.where(Y_test_labels==i)
            n_examples = len(idx[0])
            n_train = int(n_examples * train_num)
            n_val = int(n_examples * val_num)
            train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=n_val, replace=False)

            train_observed_values = observed_values[train_idx]
            val_observed_values = observed_values[val_idx]

            train_observed_masks = observed_masks[train_idx]
            val_observed_masks = observed_masks[val_idx]

            train_gt_masks = gt_masks[train_idx]
            val_gt_masks = gt_masks[val_idx]

            train_observed_pt = observed_pt[train_idx]
            val_observed_pt = observed_pt[val_idx]

            train_dataset = TensorDataset(torch.from_numpy(train_observed_values), torch.from_numpy(train_observed_masks),
                                          torch.from_numpy(train_gt_masks), torch.from_numpy(train_observed_pt))
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            train_dataloaders.append(train_dataloader)

            val_dataset = TensorDataset(torch.from_numpy(val_observed_values), torch.from_numpy(val_observed_masks),
                                        torch.from_numpy(val_gt_masks), torch.from_numpy(val_observed_pt))
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            val_dataloaders.append(val_dataloader)

        return train_dataloaders, val_dataloaders

