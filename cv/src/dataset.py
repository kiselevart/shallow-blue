import torch
import torch.nn as nn
import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

from board_to_fen import split_into_squares, filename_to_fen, fen_to_labels

class FENChessSquareDataset(Dataset):
    """
    PyTorch Dataset for loading individual chessboard square images
    and their labels derived from FEN-encoded filenames.

    Each board image file should be named with the FEN piece placement
    string, using '-' as rank separators, e.g.
    'rrQb2k1-q3R3-PR6-8-8-K3r3-p7-n2N2N1.jpeg'.

    The dataset yields (square_tensor, label_index) pairs for all 64 squares per board.
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        # accept common image extensions
        valid_exts = ('.jpeg', '.jpg', '.png')
        self.filenames = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(valid_exts)
        ]
        if not self.filenames:
            raise RuntimeError(f"No image files found in {img_dir!r}."
                               " Check path and extensions.")
        # default transform: resize to 64x64, convert to tensor
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        # map piece symbols to class indices
        self.piece_map = {
            '.': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        }

    def __len__(self):
        # 64 squares per board image
        return len(self.filenames) * 64

    def __getitem__(self, idx):
        # determine which board and which square
        board_idx = idx // 64
        square_idx = idx % 64

        filename = self.filenames[board_idx]
        img_path = os.path.join(self.img_dir, filename)
        # read and preprocess board image
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        # ensure consistent board size
        board_img = cv2.resize(img, (800, 800))
        # split into 64 square crops
        squares = split_into_squares(board_img)

        # derive FEN and convert to 64 labels
        fen = filename_to_fen(filename)
        labels = fen_to_labels(fen)

        # select the square and its label
        square_img = squares[square_idx]
        piece_symbol = labels[square_idx]
        label_idx = self.piece_map[piece_symbol]

        # apply transform
        square_tensor = self.transform(square_img)
        return square_tensor, label_idx

class ChessPieceCNN(nn.Module):
    """
    Convolutional Neural Network for classifying individual chessboard squares into
    one of 13 classes: 12 piece types (P,N,B,R,Q,K,p,n,b,r,q,k) and empty (.)
    """
    def __init__(self, num_classes: int = 13):
        super(ChessPieceCNN, self).__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 64x64 -> 32x32

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 16 * 16, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features then classify.

        Args:
            x: input tensor of shape (B, 3, 64, 64)
        Returns:
            logits tensor of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
