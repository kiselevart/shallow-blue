from position_constants import *
from bitboard import is_set

class Position:
    def __init__(self):
        self.board = ['.'] * 64

        # Piece-specific bitboards
        self.white_pawns   = START_WHITE_PAWNS
        self.white_knights = START_WHITE_KNIGHTS
        self.white_bishops = START_WHITE_BISHOPS
        self.white_rooks   = START_WHITE_ROOKS
        self.white_queens  = START_WHITE_QUEENS
        self.white_king    = START_WHITE_KING

        self.black_pawns   = START_BLACK_PAWNS
        self.black_knights = START_BLACK_KNIGHTS
        self.black_bishops = START_BLACK_BISHOPS
        self.black_rooks   = START_BLACK_ROOKS
        self.black_queens  = START_BLACK_QUEENS
        self.black_king    = START_BLACK_KING

        self.side_to_move = 'w'  # start with white

        self.update_aggregate_bitboards()

    def update_aggregate_bitboards(self):
        self.white_pieces = (
            self.white_pawns | self.white_knights | self.white_bishops |
            self.white_rooks | self.white_queens | self.white_king
        )
        self.black_pieces = (
            self.black_pawns | self.black_knights | self.black_bishops |
            self.black_rooks | self.black_queens | self.black_king
        )
        self.occupied = self.white_pieces | self.black_pieces

    def sync_board_from_bitboards(self):
        piece_map = [
            (self.white_pawns,   'P'),
            (self.white_knights, 'N'),
            (self.white_bishops, 'B'),
            (self.white_rooks,   'R'),
            (self.white_queens,  'Q'),
            (self.white_king,    'K'),
            (self.black_pawns,   'p'),
            (self.black_knights, 'n'),
            (self.black_bishops, 'b'),
            (self.black_rooks,   'r'),
            (self.black_queens,  'q'),
            (self.black_king,    'k'),
        ]

        for bitboard, symbol in piece_map:
            for sq in range(64):
                if is_set(bitboard, sq):
                    self.board[sq] = symbol

    def print_board(self):
        print("  +------------------------+")
        for rank in range(7, -1, -1):
            row = self.board[rank * 8: (rank + 1) * 8]
            print(f"{rank + 1} | {' '.join(row)} |")
        print("  +------------------------+")
        print("    a b c d e f g h")

if __name__ == "__main__":
    pos = Position()
    pos.sync_board_from_bitboards()
    pos.print_board()