import numpy as np
import torch
from torch import nn
from chess import get_data
import time

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(64 * 13 + 1, 512),
      nn.ReLU(),
      nn.Linear(512, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, 8),
      nn.ReLU(),
      nn.Linear(8, 1),
      # nn.Sigmoid()
    )

  def forward(self, x):
    return self.layers(x)

device = torch.device("cpu")

torch.manual_seed(43)
mlp = MLP().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.1)

param_count = 0
for p in mlp.parameters():
    param_count += np.prod(p.size())
print(f'total params={param_count}')

load_begin = time.time()
evals, squares, turns = get_data()
print(f"get_data() took {time.time()-load_begin:.0f}s")


for epoch in range(0, 10):
    print(f"====== starting epoch {epoch}")
    cnt = 0
    loss_sum = 0
    batch_size = 500
    last_print = time.time()
    for i in range(0, len(evals), batch_size):
        _s = torch.tensor(np.array(squares[i:i+batch_size]), dtype=torch.int64) # [B, 64]
        _s = torch.nn.functional.one_hot(_s, num_classes=13) # [B, 64, 13]
        _s = torch.flatten(_s, 1).type(torch.float32) # [B, 832]
        _t = torch.tensor(turns[i:i+batch_size]).unsqueeze(1) # [B, 1]
        xs = torch.cat((_s, _t), dim=1).to(device) # [B, 832+1]
        ys = torch.tensor(evals[i:i+batch_size]).unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = mlp(xs)
        loss = loss_function(outputs, ys)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1
        if cnt == 200:
            now = time.time()
            ns_per_board = (now - last_print) / (batch_size*cnt) * 1e9
            last_print = now
            print(f'loss={loss_sum/(cnt):.5f}\ttpb={ns_per_board:.0f}ns\tboards={i+batch_size}')
            loss_sum = 0
            cnt = 0
