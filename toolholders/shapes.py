from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence
import numpy as np
from PIL.ImageFont import ImageFont, FreeTypeFont, truetype
from skimage.measure import find_contours
from PIL import ImageDraw, Image

from toolholders.geometry import approximate_all
from .bbox import BBox, bbox_offset, bbox_unions, polygon_bbox
from numpy.typing import NDArray
from rdp import rdp_iter

class AffineTransform:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def apply(self, point: np.ndarray | tuple[float,float]) -> np.ndarray:
        if isinstance(point, tuple):
            point = np.array(point)
        # Append 1 to the end, also handle if points is an array of points
        if len(point.shape) == 1:
            assert point.shape[0] == 2
            point = np.append(point, 1)
            return np.dot(self.matrix, point)[0:2]
        else:
            assert len(point.shape) == 2 and point.shape[1] == 2
            point1 = np.column_stack([point, np.ones(point.shape[0])])
            res = np.dot(self.matrix, point1.T).T[:, 0:2]
            assert res.shape == point.shape, f"{res.shape} != {point.shape}"
            return res
    
    def apply_vector(self, vector: np.ndarray) -> np.ndarray:
        assert vector.shape[0] == 2
        vector = np.append(vector, 0)
        return np.dot(self.matrix, vector)
    
    def apply_bbox(self, bbox: BBox) -> BBox:
        # See https://zeux.io/2010/10/17/aabb-from-obb-with-component-wise-abs/
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1])
        extents = np.array([(bbox[2] - bbox[0]) / 2, (bbox[3] - bbox[1]) / 2, 0])
        center = np.dot(self.matrix, center)
        extents = np.dot(np.abs(self.matrix), extents)
        bbox = (center[0] - extents[0], center[1] - extents[1], center[0] + extents[0], center[1] + extents[1])
        return bbox
    
    def rotation(self) -> float:
        return math.atan2(self.matrix[1, 0], self.matrix[0, 0])

    def __mul__(self, other: "AffineTransform") -> "AffineTransform":
        return AffineTransform(np.dot(self.matrix, other.matrix))

    @staticmethod
    def identity() -> "AffineTransform":
        return AffineTransform(np.eye(3))

    @staticmethod
    def translate(p: NDArray | tuple[float, float]) -> "AffineTransform":
        if isinstance(p, np.ndarray):
            assert len(p.shape) == 1 and p.shape[0] == 2
        return AffineTransform(np.array([[1, 0, p[0]], [0, 1, p[1]], [0, 0, 1]]))

    @staticmethod
    def rotate(angle: float) -> "AffineTransform":
        return AffineTransform(np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]))
    
    @staticmethod
    def rotate_around_point(angle: float, point: np.ndarray) -> "AffineTransform":
        return AffineTransform.translate(point) * AffineTransform.rotate(angle) * AffineTransform.translate(-point)

class ShapeType(Enum):
    Cut = 0
    EngraveOutline = 1,
    EngraveFill = 2

@dataclass
class Shape:
    contour: np.ndarray
    type: ShapeType

    def rotated(self, angle: float) -> "Shape":
        return Shape(np.dot(self.contour, np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])), self.type)

    def bbox(self) -> BBox:
        return polygon_bbox(self.contour)

scaled_font_cache = {}

@dataclass
class Text:
    text: str
    font: FreeTypeFont
    transform: AffineTransform
    type: ShapeType
    anchor: str

    def rotated(self, angle: float) -> "Text":
        return Text(self.text, self.font, self.transform * AffineTransform.rotate(angle), self.type, self.anchor)
    
    def to_curves(self, max_error: float) -> "Group":
        resolution_mul = 20
        if self.font not in scaled_font_cache:
            scaled_font_cache[self.font] = truetype(self.font.path, self.font.size * resolution_mul)
        scaled_font = scaled_font_cache[self.font]
        bbox = scaled_font.getbbox(self.text)
        padding = 2
        offset = (-bbox[0] + padding, -bbox[1] + padding)
        buffer = np.zeros((int(bbox[3] - bbox[1]) + 2*padding, int(bbox[2] - bbox[0]) + 2*padding), dtype=np.uint8)
        im = Image.fromarray(buffer)
        context = ImageDraw.Draw(im)
        context.text(offset, self.text, font=scaled_font, fill=255)
        buffer = np.array(im)
        contours = approximate_all(find_contours(buffer), max_error * resolution_mul)
        desired_bbox = self.font.getbbox(self.text, anchor=self.anchor)
        for i in range(0, len(contours)):
            contour = contours[i]
            contour -= np.array([padding, padding])
            contour /= resolution_mul
            contour += np.array([desired_bbox[1], desired_bbox[0]])
            contour = self.transform.apply(contour)
            contours[i] = contour

        shapes = Group([Shape(contour, self.type) for contour in contours])
        return shapes

    def bbox(self) -> BBox:
        bbox = self.font.getbbox(self.text, anchor=self.anchor)
        return self.transform.apply_bbox(bbox)


@dataclass
class Group:
    children: list["Shape | Group | Text"]

    def rotated(self, angle: float) -> "Group":
        return Group([child.rotated(angle) for child in self.children])

    def bbox(self) -> BBox:
        return bbox_unions([child.bbox() for child in self.children])
