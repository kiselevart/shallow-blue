import numpy as np

def read_jpeg(img_path):
    f = open(img_path, "rb")
    data = f.read()
    hex_list = ["{:02x}".format(c) for c in data]
    print(hex_list)
    return 

def RBG_to_YCbCr(pixel_map):
    return

if __name__ == "__main__":
    file = "/Users/kisel/projects/shallow-blue/my_cv/im1.jpeg"
    print(read_jpeg(file))