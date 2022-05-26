import torch
import torch.nn.functional as F
from pdb import set_trace as bp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


device = torch.device("cuda")


d1 = torch.randn(100, 100, requires_grad=True).to(device)
d2 = torch.randn(100, 100, requires_grad=True).to(device)


base_loss = torch.tensor(0)
losses = []
for i in range(4):
    losses.append(F.mse_loss(d1, d2, reduction='none'))
bp()

loss = torch.mean(torch.cat(losses))

loss = loss + base_loss
loss.backward()

torch.cuda.empty_cache()

