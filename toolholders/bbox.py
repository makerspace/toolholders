from typing import overload
import numpy as np


BBox = tuple[float, float, float, float]
BBoxInt = tuple[int, int, int, int]

@overload
def bbox_union(bbox1: BBoxInt, bbox2: BBoxInt) -> BBoxInt: ...

@overload
def bbox_union(bbox1: BBox, bbox2: BBox) -> BBox: ...

def bbox_union(bbox1: BBox, bbox2: BBox) -> BBox:
    return (min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3]))

def bbox_unions(bboxes: list[BBox]) -> BBox:
    assert len(bboxes) > 0
    union = bboxes[0]
    for bbox in bboxes[1:]:
        union = bbox_union(union, bbox)
    return union

def bbox_area(bbox: BBox) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def polygon_bbox(polygon: np.typing.NDArray) -> BBox:
    return (polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max())

def bbox_relative_to(bbox: BBox, relative_to: BBox) -> BBox:
    return (bbox[0] - relative_to[0], bbox[1] - relative_to[1], bbox[2] - relative_to[0], bbox[3] - relative_to[1])

def bbox_center(bbox: BBox | BBoxInt) -> tuple[float,float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

@overload
def bbox_grow(bbox: BBoxInt, padding: int) -> BBoxInt: ...

@overload
def bbox_grow(bbox: BBox, padding: int) -> BBox: ...

def bbox_grow(bbox: BBox, padding: float) -> BBox:
    return (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)

@overload
def bbox_offset(bbox: BBoxInt, offset: tuple[int,int]) -> BBoxInt: ...

@overload
def bbox_offset(bbox: BBox, offset: tuple[float,float]) -> BBox: ...

def bbox_offset(bbox: BBox, offset: tuple[float,float]) -> BBox:
    return (bbox[0] + offset[0], bbox[1] + offset[1], bbox[2] + offset[0], bbox[3] + offset[1])