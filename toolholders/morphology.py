import skimage.morphology as morphology
import numpy as np
import math
from skimage.measure import regionprops
from PIL import Image
from numpy.typing import NDArray
from toolholders.bbox import BBox, BBoxInt

def dilate_direction(mask, direction, distance):
    if direction == "down":
        direction = np.concat([np.zeros((distance, 1)), np.ones((distance, 1))])
    elif direction == "up":
        direction = np.concat([np.ones((distance, 1)), np.zeros((distance, 1))])
    elif direction == "right":
        direction = np.concat([np.zeros((1, distance)), np.ones((1, distance))], axis=1)
    elif direction == "left":
        direction = np.concat([np.ones((1, distance)), np.zeros((1, distance))], axis=1)
    return morphology.binary_dilation(mask, direction)

def triangle_footprint(distance, angle):
    footprint = np.zeros((2*distance + 1, 2*distance + 1))
    for i in range(2*distance+1):
        for j in range(2*distance+1):
            if abs(math.atan2(j - distance, i - distance)) < angle:
                footprint[i, j] = 1
    
    # # Flip upside down
    # footprint = np.flipud(footprint)
    # # Mirror left to right and concat
    # footprint = np.concatenate([np.fliplr(footprint), footprint], axis=1)
    # # Center, by concatenating with empty rows
    # footprint = np.concatenate([footprint, np.zeros((distance, 2*distance))])
    return footprint

def crop_to_bbox(mask, padding: int):
    props = regionprops(mask.astype(np.uint8))
    bbox = props[0].bbox
    # Crop and handle padding and out of bounds
    bbox = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    def f(img):
        assert mask.shape[0:2] == img.shape[0:2], f"{mask.shape[0:2]} != {img.shape[0:2]}"
        return crop_to_fit_bbox(img, bbox)
    return f

def crop_to_fit_bbox(source_img: NDArray | Image.Image, bbox: BBoxInt):
    if isinstance(source_img, Image.Image):
        img = np.array(source_img)
    else:
        img = source_img

    # Pad with zeros if bbox is out of bounds
    padding_values = ((max(0, -bbox[0]), max(0, bbox[2] - img.shape[0])), ((max(0, -bbox[1]), max(0, bbox[3] - img.shape[1]))))
    if len(img.shape) == 3:
        padding_values += ((0, 0),)
    
    shift = (padding_values[0][0], padding_values[1][0])
    bbox_offset = (bbox[0] + shift[0], bbox[1] + shift[1], bbox[2] + shift[0], bbox[3] + shift[1])
    img = np.pad(img, padding_values)

    result = img[bbox_offset[0]:bbox_offset[2], bbox_offset[1]:bbox_offset[3]]
    return result