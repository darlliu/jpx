import torch
import torch.nn as nn
import torch.nn.functional as F


class StockNet(nn.Module):
    """
    Simple first attempt at a model
    1. embedding on stock symbol. e.g. 4500 x 16
    2. input vector is [emb_idx, start, close, high, low, vol, adj]
    3. turns into [emb] ++ [prices] (concat)
    4. (hidden/CNN)
    5. output -> pairwise loss construction 
    """

    def __init__(self, emb_size=4500, emb_dim=16, ndim=6):
        super(StockNet, self).__init__()
        self.encoder = nn.Embedding(emb_size, emb_dim)
        self.nn1 = nn.Linear(emb_dim + ndim, 64)
        self.nn2 = nn.Linear(64, 32)
        self.nn3 = nn.Linear(32, 16)
        self.nout = nn.Linear(16, 1)
        self.init_weights()
        return

    def init_weights(self):
        rr = 0.1
        self.encoder.weight.data.uniform_(-rr, rr)
        return

    def forward(self, ids, xs):
        # src is 1 + 6, nbatch sized
        emb = self.encoder(ids)
        x = torch.cat([torch.squeeze(emb), xs], 1)
        x1 = F.tanh(self.nn1(x))
        x2 = F.tanh(self.nn2(x1))
        x3 = F.tanh(self.nn3(x2))
        output = self.nout(x3)
        return F.tanh(output)


if __name__ == "__main__":
    net = StockNet()
    print(net)
