def set_bit(bb: int, square: int) -> int:
    return bb | (1 << square)

def clear_bit(bb: int, square: int) -> int:
    return bb & (0 << square)

def toggle_bit(bb: int, square: int) -> int:
    return bb ^ (1 << square)

def is_set(bb: int, square: int) -> bool:
    return bool((bb >> square) & 1)

def popcount(bb: int) -> int:
    return bb.bit_count()
    
def bitscan_forward(bb: int ) -> int:
    if bb == 0:
        raise ValueError("Cannot scan empty bitboard")
    return (bb & -bb).bit_length() -1

def bitscan_backward(bb: int ) -> int:
    if bb == 0:
        raise ValueError("Cannot scan empty bitboard")
    return bb.bit_length() -1

def print_bitboard(bb: int) -> None:
    print("Bitboard:")
    for rank in reversed(range(8)):
        row = ""
        for file in range(8):
            sq = rank * 8 + file
            row += " 1 " if is_set(bb, sq) else " . "
        print(row)
    print()

def square_index(file: str, rank: int) -> int:
    return (rank - 1) * 8 + "abcdefgh".index(file)

def index_to_square(index: int):
    print(index)
    rank = index % 8 
    file = index - rank*8
    sq = f"{"abcdefgh"[file]}{rank+1}"
    return sq

if __name__ == "__main__":
    rank1 = [
        ('a',1),            
        ('b',1),            
        ('c',1),            
        ('d',1),            
        ('e',1),            
        ('f',1),            
        ('g',1),            
        ('h',1),            
    ]
    bb = 0
    for r in rank1:
        print(r)
        bb = set_bit(bb, square_index(r[0], r[1]))
    print(bin(bb))

    print_bitboard(bb)
    print(bin(bb))
    print(hex(bb))
    print(index_to_square(63))