#!/usr/bin/env python3

import numpy as np
from PIL import Image
import colorsys


def create_test_grid(size=201, grid_spacing=20):
    """
    Create test image with:
    - Full HSV gradient clockwise from bottom: Red -> Yellow -> Green -> Cyan -> Blue -> Magenta -> Red
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            cx = (x - size // 2) / (size // 2)
            cy = (y - size // 2) / (size // 2)

            angle = np.arctan2(-cy, cx)
            angle_deg = np.degrees(angle)

            dist = min(np.sqrt(cx ** 2 + cy ** 2), 1.0)

            hue = (270 - angle_deg) % 360

            h = hue / 360.0
            s = 0.7 + 0.3 * dist
            v = 0.85

            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            img[y, x] = [int(r * 255), int(g * 255), int(b * 255)]

    for i in range(0, size, grid_spacing):
        img[:, i] = [0, 0, 0]
        img[i, :] = [0, 0, 0]

    center = size // 2
    img[center - 1:center + 2, :] = [255, 255, 255]
    img[:, center - 1:center + 2] = [255, 255, 255]

    img[0, :] = [255, 255, 255]
    img[-1, :] = [255, 255, 255]
    img[:, 0] = [255, 255, 255]
    img[:, -1] = [255, 255, 255]

    return Image.fromarray(img, 'RGB')

def main():
    test_img = create_test_grid(size=1025, grid_spacing=20)
    test_img.save("test_grid.png")
    print("Created test_grid.png (201x201, 20px grid)")

if __name__ == "__main__":
    main()
