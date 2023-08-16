import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

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


def get_data(limit=None):
    with open("chessData.csv") as f:
        data = f.readlines()
    print(f"Loaded {len(data)} examples")

    count = len(data[:limit])

    already_parsed = Path("evals.bin").exists()
    if not already_parsed:
        Path("evals.bin").touch()
        Path("squares.bin").touch()
        Path("turns.bin").touch()
        print("Seems like chessData.csv has not been parsed alredy, parsing...")

    evals = np.memmap("evals.bin", dtype=np.float32, shape=(count, ))
    squares = np.memmap("squares.bin", dtype=np.int8, shape=(count, 64))
    turns = np.memmap("turns.bin", dtype=np.int8, shape=(count,))

    if not already_parsed:
        for idx, line in enumerate(tqdm(data[:limit])):
            fen, evaluation = line.split(',')

            # Parse evaluation
            evaluation = evaluation.strip()
            if evaluation[0] == '#': evaluation = evaluation[1:] + '000'
            evaluation = float(evaluation) / 100
            if evaluation > 1.0: evaluation = 1
            if evaluation < -1.0: evaluation = -1
            evals[idx] = evaluation

            # Parse the fen
            s, turn = parse_fen(fen)
            squares[idx] = s
            turns[idx] = turn
        
        evals.flush()
        squares.flush()
        turns.flush()

    return evals, squares, turns
