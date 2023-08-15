import torch
import numpy as np
from torch import nn
import time

# with open("cd.csv") as f:
with open("chessData.csv") as f:
    data = f.readlines()

print(f"Loaded {len(data)} examples")

def parse_board(board: str):
    squares = np.zeros(64)
    char2sqaure = {
        #  0 empty
        'K':  1, #  1 white king
        'Q':  2, #  2 white queen
        'R':  3, #  3 white rook
        'B':  4, #  4 white bishop
        'N':  5, #  5 white knight
        'P':  6, #  6 white pawn
        'k':  7, #  7 black king
        'q':  8, #  8 black queen
        'r':  9, #  9 black rook
        'b': 10, # 10 black bishop
        'n': 11, # 11 black knight
        'p': 12, # 12 black pawn
    }

    idx = 0
    for c in board:
        if c == '/': continue
        elif c == '1': idx += 1
        elif c == '2': idx += 2
        elif c == '3': idx += 3
        elif c == '4': idx += 4
        elif c == '5': idx += 5
        elif c == '6': idx += 6
        elif c == '7': idx += 7
        elif c == '8': idx += 8
        else:
            squares[idx] = char2sqaure[c]
            idx += 1
    
    return squares


def parse_fen(fen: str):
    board, active, castling, enpassant, halfmove, fullmove = fen.split(' ')
    turn = 1 if active == 'w' else -1
    return parse_board(board), turn

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 13 + 1, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, 8),
      nn.ReLU(),
      nn.Linear(8, 1)
    )

  def forward(self, x):
    return self.layers(x)

torch.manual_seed(42)
mlp = MLP()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)

param_count = 0
for p in mlp.parameters():
    param_count += np.prod(p.size())
print(f'total params={param_count}')
sys.exit(0)

for epoch in range(0, 10):
    print(f"====== starting epoch {epoch}")
    cnt = 0
    loss_sum = 0
    batch_size = 400
    last_print = time.time()
    for i in range(0, len(data), batch_size):
        ys = []
        squares = []
        turns = []
        for sample in data[i:i+batch_size]:
            fen, sfeval = sample.split(',')

            # Parse eval
            sfeval = sfeval.strip()
            if sfeval[0] == '#': sfeval = sfeval[1:] + '000'
            sfeval = float(sfeval) / 100
            if sfeval > 1.0: sfeval = 1
            if sfeval < -1.0: sfeval = -1
            # print(sfeval)
            # if sfeval > 999: sfeval = 999

            ys.append(sfeval)
            s, turn = parse_fen(fen)
            squares.append(s)
            turns.append(turn)

        _s = torch.tensor(np.array(squares), dtype=torch.int64) # [B, 64]
        _s = torch.nn.functional.one_hot(_s, num_classes=13) # [B, 64, 13]
        _s = torch.flatten(_s, 1).float() # [B, 832]
        _t = torch.tensor(turns).unsqueeze(1) # [B, 1]
        xs = torch.cat((_s, _t), dim=1) # [B, 832+1]

        ys = torch.tensor(ys).unsqueeze(1)

        optimizer.zero_grad()
        outputs = mlp(xs)
        loss = loss_function(outputs, ys)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        # print(loss.item())
        cnt += 1
        if cnt == 200:
            now = time.time()
            ns_per_board = (now - last_print) / (batch_size*cnt) * 1e6
            last_print = now
            print(f'loss={loss_sum/(cnt):.5f}\ttime={ns_per_board:.1f}ns\tboards={i+batch_size}')
            loss_sum = 0
            cnt = 0
        # break


# current_loss=0
# for idx, sample in enumerate(data):
# 
# 
#     # Print statistics
#     current_loss += loss.item()
#     if idx % 500 == 499:
#         print(f'Loss after mini-batch {idx+1}: {current_loss/500:.0f}')
#         current_loss = 0.0
# 
#     # print(f'{idx} done')
