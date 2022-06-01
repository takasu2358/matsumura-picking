import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import ply2depth_csv

def expansion_image(image):
    left_merge, top_merge = 100, 60
    right_merge, bottom_merge = 50, 0
    height, width = image.shape

    pil_image = Image.fromarray(image)

    pil_image = pil_image.crop((left_merge, top_merge, width-right_merge, height-bottom_merge))
    height, width = pil_image.size

    pil_image = pil_image.resize((height*2, width*2))
    image = np.array(pil_image)

    return image

def main(filepath):
    image = ply2depth_csv.main(filepath)
    image = expansion_image(image)
    cv2.imwrite("/home/takasu/ダウンロード/matsumura-picking-git/DEPTH_TMP/reshape.png", image)
    return image

if __name__ == "__main__":
    # image = cv2.imread("./data/test/depthimg.png", 0)
    filepath = "/home/takasu/ダウンロード/matsumura-picking-git/input/out.ply"
    image = ply2depth_csv.main(filepath)
    image = expansion_image(image)
    cv2.imwrite("/home/takasu/ダウンロード/matsumura-picking-git/DEPTH_TMP/reshape.png", image)
