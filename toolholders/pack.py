from dataclasses import replace
from .bbox import BBox, bbox_area, polygon_bbox
from .shapes import AffineTransform, Shape, Group, Text
from typing import Sequence
from rectpack import newPacker
import numpy as np
import math

def rotate_shape_to_minimize_bbox(shape: Shape | Group):
    best_score = float('inf')
    best_angle = 0.0
    for a in np.linspace(0, 180, 180//10):
        rotated = shape.rotated(math.radians(a))
        bbox = rotated.bbox()
        score = bbox_area(bbox)
        if score < best_score:
            best_score = score
            best_angle = a
    
    for a in np.linspace(best_angle-10, best_angle+10, 10):
        rotated = shape.rotated(math.radians(a))
        bbox = rotated.bbox()
        score = bbox_area(bbox)
        if score < best_score:
            best_score = score
            best_angle = a
    
    return shape.rotated(math.radians(best_angle))

def pack_shapes(shapes: Sequence[Shape | Group], margin: float):
    # Pack shapes
    # Binary search over smallest bin size that can fit all shapes
    mn = 1.0
    mx = 2.0
    last_ok_size = 0.0
    last_ok_rects = []

    def try_pack(size, shapes):
        packer = newPacker(rotation=True)
        for shape in shapes:
            bbox = shape.bbox()
            w = bbox[2] - bbox[0] + 2*margin
            h = bbox[3] - bbox[1] + 2*margin
            packer.add_rect(w, h, rid=(shape, w, h))

        packer.add_bin(size[0], size[1])
        packer.pack()
        rects = packer.rect_list()
        if len(rects) == len(shapes):
            return rects
        return None

    while True:
        rects = try_pack((mx, mx), shapes)
        if rects is not None:
            last_ok_rects = rects
            last_ok_size = mx
            break
        else:
            mn = mx
            mx *= 2
    
    while mn + 0.5 < mx:
        mid = (mn + mx) / 2
        rects = try_pack((mid, mid), shapes)
        if rects is not None:
            last_ok_rects = rects
            last_ok_size = mid
            mx = mid
        else:
            mn = mid

    # Binary search over each axis separately
    last_ok_size2 = np.array([last_ok_size, last_ok_size])
    for d in range(2):
        mn2 = last_ok_size2[d]/2
        mx2 = last_ok_size2[d]
        while mn2 + 0.5 < mx2:
            mid = (mn2 + mx2) / 2
            size = last_ok_size2.copy()
            size[d] = mid
            rects = try_pack((size[0], size[1]), shapes)
            if rects is not None:
                last_ok_rects = rects
                last_ok_size2 = size
                mx2 = mid
            else:
                mn2 = mid
    
    return last_ok_size2, last_ok_rects


def move_packed_shapes(rects, packing_margin: float):
    output = []
    for rect in rects:
        bin, x, y, w, h, (shape, orig_w, orig_h) = rect
        rotated = w != orig_w
        original_bbox = shape.bbox()

        
        rotated_bbox = original_bbox
        # if rotated:
        #     rotated_bbox = (original_bbox[1], original_bbox[0], original_bbox[3], original_bbox[2])
        current_xy = rotated_bbox[0:2]
        desired_xy = np.array([x + packing_margin, y + packing_margin])
        offset = desired_xy - current_xy

        def move_shape(shape: Shape | Text):
            if isinstance(shape, Text):
                new_transform = shape.transform
                
                if rotated:
                    new_transform = (
                        new_transform
                        * AffineTransform.translate((0, original_bbox[2] - original_bbox[2]))
                        * AffineTransform.rotate_around_point(-math.pi/2, np.array(original_bbox[0:2]))
                    )
                new_transform = new_transform * AffineTransform.translate(offset)
                return replace(shape, transform=new_transform)
            else:
                contour = shape.contour
                if rotated:
                    contour -= np.array(original_bbox[0:2])
                    contour = np.array([contour[:, 1], -contour[:, 0] + (original_bbox[2] - original_bbox[0])]).T
                    contour += np.array(original_bbox[0:2])
                contour += offset
                return Shape(contour, shape.type)
        
        def move_item(item: Shape | Group | Text):
            if isinstance(item, Shape) or isinstance(item, Text):
                return move_shape(item)
            else:
                return Group([move_item(child) for child in item.children])

        shape = move_item(shape)
        output.append(shape)
    return output