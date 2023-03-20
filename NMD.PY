import numpy as np
import sys
from PIL import Image


def euclidean_distance(window1, window2):
    return np.sqrt(np.sum((window1 - window2) ** 2))


def non_local_means(image, search_window_size, similarity_window_size, h):
    height, width = image.shape
    output = np.zeros_like(image)

    sw_half = search_window_size // 2
    simw_half = similarity_window_size // 2
    for i in range(height):
        for j in range(width):
            weights = []
            pixel_value = 0.0
            for k in range(max(0, i - sw_half), min(height, i + sw_half + 1)):
                for l in range(max(0, j - sw_half), min(width, j + sw_half + 1)):
                    if i == k and j == l:
                        continue
                    i1, i2 = max(0, i - simw_half), min(height, i + simw_half + 1)
                    j1, j2 = max(0, j - simw_half), min(width, j + simw_half + 1)
                    k1, k2 = max(0, k - simw_half), min(height, k + simw_half + 1)
                    l1, l2 = max(0, l - simw_half), min(width, l + simw_half + 1)
                    window1 = image[i1:i2, j1:j2]
                    window2 = image[k1:k2, l1:l2]

                    if window1.shape == window2.shape:
                        dist = euclidean_distance(window1, window2)
                        weight = np.exp(-(dist ** 2) / (h ** 2))
                        pixel_value += weight * image[k, l]
                        weights.append(weight)
            output[i, j] = pixel_value / np.sum(weights)

    return output


def image_decomposition(image_path, search_window_size=11, similarity_window_size=3, h=10.0):
    input_image = Image.open(image_path).convert("L")
    image = np.asarray(input_image, dtype=np.float32)

    structure_layer = non_local_means(image, search_window_size, similarity_window_size, h)
    texture_layer = image - structure_layer

    return Image.fromarray(structure_layer.astype(np.uint8)), Image.fromarray(texture_layer.astype(np.uint8))


if __name__ == "__main__":
    input_image_path = '000012.png'
    structure_layer, texture_layer = image_decomposition(input_image_path)

    structure_layer.save("structure_layer.png")
    texture_layer.save("texture_layer.png")
