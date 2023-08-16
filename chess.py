import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

def parse_board(board: str):
    squares = np.zeros(64 * 6)
    char2sqaure = {
        # 'e': np.array([ 0, 0, 0, 0, 0, 0]), #    empty
        'K': np.array([+1, 0, 0, 0, 0, 0]), #  1 white king
        'Q': np.array([ 0,+1, 0, 0, 0, 0]), #  2 white queen
        'R': np.array([ 0, 0,+1, 0, 0, 0]), #  3 white rook
        'B': np.array([ 0, 0, 0,+1, 0, 0]), #  4 white bishop
        'N': np.array([ 0, 0, 0, 0,+1, 0]), #  5 white knight
        'P': np.array([ 0, 0, 0, 0, 0,+1]), #  6 white pawn
        'k': np.array([-1, 0, 0, 0, 0, 0]), #  7 black king
        'q': np.array([ 0,-1, 0, 0, 0, 0]), #  8 black queen
        'r': np.array([ 0, 0,-1, 0, 0, 0]), #  9 black rook
        'b': np.array([ 0, 0, 0,-1, 0, 0]), # 10 black bishop
        'n': np.array([ 0, 0, 0, 0,-1, 0]), # 11 black knight
        'p': np.array([ 0, 0, 0, 0, 0,-1]), # 12 black pawn
    }

    idx = 0
    for c in board:
        if c == '/':
            continue
        if '1' <= c <= '8':
            idx += ord(c) - ord('0')
            continue
        squares[idx*6:(idx+1)*6] = char2sqaure[c]
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
    # already_parsed = False
    if not already_parsed:
        Path("evals.bin").touch()
        Path("squares.bin").touch()
        Path("turns.bin").touch()
        print("Seems like chessData.csv has not been parsed alredy, parsing...")

    evals = np.memmap("evals.bin", dtype=np.float32, shape=(count, ))
    squares = np.memmap("squares.bin", dtype=np.int8, shape=(count, 64*6))
    turns = np.memmap("turns.bin", dtype=np.int8, shape=(count,))

    if not already_parsed:
        for idx, line in enumerate(tqdm(data[:limit])):
            fen, evaluation = line.split(',')

            # Parse evaluation
            evaluation = evaluation.strip()
            if evaluation[0] == '#': evaluation = evaluation[1:] + '000'
            evaluation = float(evaluation) / 1000
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
