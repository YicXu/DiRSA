import numpy as np
import pickle
import os
import pickle as cPickle

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

def normalize(X):
    std_dev = np.std(X, axis=(1, 2), keepdims=True)

    k = 1.0 / std_dev

    normalized_data = X * k

    return normalized_data

def gendata(seed=2023, read_origin=False, if_RAF_aug=False, filepath_suff=''):

    np.random.seed(seed)  # seed for ground truth choice

    pathTrain_idx = '/home/xyc/CSDI-IQ/data/DiffTrain_idx'+filepath_suff+'.npy'
    pathTest_idx = '/home/xyc/CSDI-IQ/data/DiffTest_idx'+filepath_suff+'.npy'
    pathVal_idx = '/home/xyc/CSDI-IQ/data/DiffVal_idx'+filepath_suff+'.npy'

    datafolder = 'pretrained'  # set the folder name
    path = '/home/xyc/CSDI-IQ/save/' + datafolder + '/generated_outputs_nsample' + filepath_suff + '.pk'
    if (not read_origin):
        with open(path, 'rb') as f:
            X2, X, all_evalpoint, all_observed, all_observed_time, Y, test_SNRs, scaler, mean_scaler = pickle.load(
                f)

        X2 = np.squeeze(X2, axis=1)
        X = X2 * all_evalpoint[:, :, :] + X * (1 - all_evalpoint[:, :, :])

        X = normalize(X)

    fp = '/home/xyc/AMC/RML2016.10a_dict.pkl'
    Xd = cPickle.load(open(fp, 'rb'), encoding="latin-1")
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X2 = []
    lbl = []
    print(mods, snrs)
    for mod in mods:
        for snr in snrs:
            X2.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))

    stack_X = np.vstack(X2)
    ct = np.arange(0, len(stack_X))
    temp_snrs = list(map(lambda x: int(lbl[x][1]), ct))
    temp_snrs = np.array(temp_snrs)
    stack_X = normalize(stack_X)
    stack_X = np.transpose(stack_X,[0,2,1])

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    temp_Y = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), ct)))
    temp_Y = np.array(temp_Y)

    f = 0
    if os.path.exists(pathTrain_idx) and os.path.exists(pathTest_idx) and os.path.exists(pathVal_idx):
        train_idx = np.load(pathTrain_idx)
        test_idx = np.load(pathTest_idx)
        val_idx = np.load(pathVal_idx)

    Xo = np.array(stack_X[train_idx], dtype='float32')
    Yo = temp_Y[train_idx]
    test_SNRso = temp_snrs[train_idx]
    if read_origin:
        X = np.array(stack_X[train_idx], dtype='float32')
        Y = temp_Y[train_idx]
        test_SNRs = temp_snrs[train_idx]
    else:
        X = np.concatenate((Xo, X), axis=0)
        Y = np.concatenate((Yo, Y), axis=0)
        test_SNRs = np.concatenate((test_SNRso, test_SNRs), axis=0)

    if if_RAF_aug:
        Xo = np.transpose(Xo, [0, 2, 1])
        X = np.transpose(X, [0, 2, 1])
        # Rotate
        rotate1 = np.array([[0, -1], [1, 0]])
        rotate2 = np.array([[-1, 0], [0, -1]])
        rotate3 = np.array([[0, 1], [-1, 0]])
        x1 = rotate_data(Xo, rotate1)
        x2 = rotate_data(Xo, rotate2)
        x3 = rotate_data(Xo, rotate3)

        # Flip
        flip1 = np.array([-1, 1])
        flip2 = np.array([1, -1])
        flip3 = np.array([-1, -1])
        x4 = flip_data(Xo, flip1)
        x5 = flip_data(Xo, flip2)
        x6 = flip_data(Xo, flip3)

        X = np.concatenate([X, x1, x2, x3, x4, x5, x6], axis=0)
        Y = np.concatenate([Y, Yo, Yo, Yo, Yo, Yo, Yo], axis=0)
        test_SNRs = np.concatenate([test_SNRs, test_SNRso, test_SNRso, test_SNRso, test_SNRso, test_SNRso, test_SNRso], axis=0)
        X = np.transpose(X, [0, 2, 1])

    X_test = np.array(stack_X[test_idx], dtype='float32')
    Y_test = temp_Y[test_idx]
    test_SNRs = temp_snrs[test_idx]

    X_val = np.array(stack_X[val_idx], dtype='float32')
    Y_val = temp_Y[val_idx]
    val_SNRs = temp_snrs[val_idx]

    return (X, Y, X_test, Y_test, X_val, Y_val, test_SNRs)