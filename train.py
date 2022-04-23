from data import Stock, TSDataset, LOOKUP, REVERSE_LOOKUP, DSTrain
from model import StockNet
import torch
import time
import numpy as np
import random
from pytorchltr.loss import PairwiseLogisticLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is >{device}<")

EMB_SIZE = len(LOOKUP)
EMB_DIM = 4
NDIM = 10
LEARNING_RATE = 1
NBATCH = 32

model = StockNet(EMB_SIZE, EMB_DIM, NDIM)
loss_fctn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def prep_data(rows):
    """
    turns data into pytorch tensors ready to be used
    """
    ys = [r.targets[-1] for r in rows]
    xs = [r.features() for r in rows]
    embs = [LOOKUP[r.code] for r in rows]
    return torch.nan_to_num(torch.tensor(embs, dtype=torch.int64)),\
        torch.nan_to_num(torch.tensor(xs, dtype=torch.float32)),\
        torch.nan_to_num(torch.tensor(ys, dtype=torch.float32))


def make_batch(E, X, Y):
    perms = torch.randperm(E.size()[0])
    for idx in range(0, E.size()[0], NBATCH):
        idxs = perms[idx: idx + NBATCH]
        if len(idxs) < NBATCH:
            continue
        yield torch.reshape(E[idxs], [NBATCH, 1]), torch.reshape(X[idxs], [NBATCH, NDIM]), torch.reshape(Y[idxs], [NBATCH, 1])


start_time = time.time()
total_loss = 0
cnt = 0
for ts, rows in DSTrain:
    E, X, Y = prep_data(rows)
    model.eval()
    with torch.no_grad():
        # evaluate first
        O = torch.flatten(model.forward(E, X))
        print(
            f"TS: {ts} || Total Abs Loss: {torch.sum(torch.abs(Y - O))/Y.size()[0]} || Score TBA")
    model.train()
    for ii, (e, x, y) in enumerate(make_batch(E, X, Y)):
        # now train model
        optimizer.zero_grad()
        o = model(e, x)
        # here N = 1 (one single learning instance)
        # docs = NBATCH
        # loss = loss_fctn(torch.reshape(o, [1, NBATCH]), torch.reshape(y, [1, NBATCH]),
        #                 torch.tensor([NBATCH]))
        loss = loss_fctn(o, y)
        loss.backward()
        total_loss += loss.sum()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # show training loss
        cnt += 1
    if cnt % 20 == 0 and cnt > 0:
        cur_loss = total_loss / 20
        elapsed = time.time() - start_time
        print(o, y)
        print(
            f"iteration {cnt} | speed {elapsed/200} | lr {scheduler.get_last_lr()[0]} | loss {cur_loss}")
        start_time = time.time()
        total_loss = 0
        scheduler.step()
