import cv2
import numpy as np

def img_to_fen(img_path, model):
    img = cv2.imread(img_path)
    board_img = preprocess_board(img)
    squares = split_into_squares(board_img)
    pieces = classify_squares(squares, model)
    fen = pieces_to_fen(pieces)

def preprocess_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    board = cv2.resize(gray, (800, 800))
    return board

def split_into_squares(board_img):
    h, w = board_img.shape
    squares = []
    sq_h, sq_w = h // 8, w // 8 
    for i in range(8):
        for j in range(8):
            square = board_img [
                i*sq_h:(i+1)*sq_h, j*sq_w:(j+1)*sq_w
            ]
            squares.append(square)
    return squares

def classify_squares(squares, model):
    labels = []
    for square in squares:
        square_resized = cv2.resize(square, (64, 64))
        label = model.predict(square_resized)
        labels.append(label)
    return labels