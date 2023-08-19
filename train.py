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
      nn.LeakyReLU(),
      nn.Linear(48, 24),
      nn.LeakyReLU(),
      nn.Linear(24, 16),
      nn.LeakyReLU(),
      nn.Linear(16, 8),
      nn.LeakyReLU(),
      nn.Linear(8, 1),
      # nn.Sigmoid()
    )

  def forward(self, x):
    return self.layers(x)

def encode(squares, turns):
    _s = torch.tensor(squares).float() # [B, 64*6]
    # _s = torch.nn.functional.one_hot(_s, num_classes=13) # [B, 64, 13]
    # _s = torch.flatten(_s, 1).type(torch.float32) # [B, 832]
    _t = torch.tensor(turns).unsqueeze(1) # [B, 1]
    return torch.cat((_s, _t), dim=1) # [B, 832+1]

# def val(mlp):
#     # https://lichess.org/Hj7N23o8/black#69 - +6.3 s1, t1 = parse_fen('1R6/p3k3/2p5/8/3b1rN1/5NK1/2Q3PP/q7 b - - 8 35')
#
#     # https://lichess.org/uZZ7jtZP/white#84 - -8.4
#     s2, t2 = parse_fen('4R3/kp6/p7/6pp/1r1q4/P6P/8/7K w - - 2 43')
#
#     # https://lichess.org/R9Zq7235/white#26 - +9999 mate in 1
#     s3, t3 = parse_fen('r1b1r1k1/pppp1ppp/1nn5/1B6/8/5N2/PP1N1PPP/4RRK1 w - - 2 14')
#
#     # print(s1)
#     xs = encode(np.array([s1, s2, s3]), np.array([t1, t2, t3])).to("mps")
#     # print(xs[0])
#     with torch.no_grad():
#         preds = mlp(xs)
#         print(preds)


if __name__ == '__main__':
    # device = torch.device("mps")
    device = torch.device("cpu")

    torch.manual_seed(43)
    mlp = MLP().to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    param_count = 0
    for p in mlp.parameters():
        param_count += np.prod(p.size())
    print(f'total params={param_count}')

    load_begin = time.time()
    all_evals, all_squares, all_turns = get_data()
    print(f"get_data() took {time.time()-load_begin:.2f}s")

    test_evals    = all_evals[:50000]
    test_squares  = all_squares[:50000]
    test_turns    = all_turns[:50000]
    train_evals   = all_evals[50000:]
    train_squares = all_squares[50000:]
    train_turns   = all_turns[50000:]

    for epoch in range(0, 100):
        with torch.no_grad():
            xs = encode(test_squares, test_turns).to(device)
            ys = torch.tensor(test_evals).unsqueeze(1).to(device)
            outputs = mlp(xs)
            loss = loss_function(outputs, ys)
            print(f"====== starting epoch {epoch} test_loss={loss.item()}")

        cnt = 0
        loss_sum = 0
        batch_size = 256
        last_print = time.time()
        for i in range(0, len(train_evals), batch_size):
            xs = encode(train_squares[i:i+batch_size], train_turns[i:i+batch_size]).to(device)
            ys = torch.tensor(train_evals[i:i+batch_size]).unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = mlp(xs)
            loss = loss_function(outputs, ys)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1
            if cnt == 10000:
                now = time.time()
                ns_per_board = (now - last_print) / (batch_size*cnt) * 1e9
                last_print = now
                print(f'train loss={loss_sum/(cnt):.5f}\ttpb={ns_per_board:.0f}ns\tboards={i+batch_size}')
                loss_sum = 0
                cnt = 0

