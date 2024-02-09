import torch
import torch.nn.functional as F
import torch.nn as nn

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class LSTMCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.out = nn.Linear(in_features=128, out_features=11)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output_in_last_timestep = h_n[-1, :, :]
        x = self.out(output_in_last_timestep)
        x = F.log_softmax(x, dim=1)
        return x

