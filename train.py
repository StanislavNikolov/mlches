import numpy as np
import torch
from torch import nn
from chess import get_data, parse_fen
import time

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(64 * 6 + 1, 48),
      nn.ReLU(),
      nn.Linear(48, 24),
      nn.ReLU(),
      nn.Linear(24, 16),
      nn.ReLU(),
      nn.Linear(16, 8),
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
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

param_count = 0
for p in mlp.parameters():
    param_count += np.prod(p.size())
print(f'total params={param_count}')

load_begin = time.time()
evals, squares, turns = get_data()
print(f"get_data() took {time.time()-load_begin:.0f}s")

def encode(squares, turns):
    _s = torch.tensor(squares).float() # [B, 64*6]
    # _s = torch.nn.functional.one_hot(_s, num_classes=13) # [B, 64, 13]
    # _s = torch.flatten(_s, 1).type(torch.float32) # [B, 832]
    _t = torch.tensor(turns).unsqueeze(1) # [B, 1]
    return torch.cat((_s, _t), dim=1).to(device) # [B, 832+1]

def val():
    # https://lichess.org/Hj7N23o8/black#69 - +6.3
    s1, t1 = parse_fen('1R6/p3k3/2p5/8/3b1rN1/5NK1/2Q3PP/q7 b - - 8 35')

    # https://lichess.org/uZZ7jtZP/white#84 - -8.4
    s2, t2 = parse_fen('4R3/kp6/p7/6pp/1r1q4/P6P/8/7K w - - 2 43')

    # https://lichess.org/R9Zq7235/white#26 - +9999 mate in 1
    s3, t3 = parse_fen('r1b1r1k1/pppp1ppp/1nn5/1B6/8/5N2/PP1N1PPP/4RRK1 w - - 2 14')


    # print(s1)
    xs = encode(np.array([s1, s2, s3]), np.array([t1, t2, t3]))
    # print(xs[0])
    with torch.no_grad():
        preds = mlp(xs)
        print(preds)


for epoch in range(0, 50):
    print(f"====== starting epoch {epoch}")
    cnt = 0
    loss_sum = 0
    batch_size = 1000
    last_print = time.time()
    for i in range(0, len(evals), batch_size):
        xs = encode(squares[i:i+batch_size], turns[i:i+batch_size])
        ys = torch.tensor(evals[i:i+batch_size]).unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = mlp(xs)
        loss = loss_function(outputs, ys)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1
        if cnt == 1000:
            now = time.time()
            ns_per_board = (now - last_print) / (batch_size*cnt) * 1e9
            last_print = now
            print(f'loss={loss_sum/(cnt):.5f}\ttpb={ns_per_board:.0f}ns\tboards={i+batch_size}')
            loss_sum = 0
            cnt = 0

            # val()
