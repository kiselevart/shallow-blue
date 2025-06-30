from PIL import Image
import os 

class FENChessSquareDataset:
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        file_extension = '.jpeg'

        self.filenames = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(file_extension)
        ]
        if not self.filenames:
            raise RuntimeError(f"No image files found in {img_dir!r}.")
        
        self.transform = transform or tra