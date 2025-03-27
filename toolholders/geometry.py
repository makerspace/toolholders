import math
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from skimage.measure import points_in_poly
from rdp import rdp_iter

def circle(center: NDArray, radius: float, resolution: int) -> NDArray:
    angles = np.linspace(0, 2*math.pi, resolution)
    return np.array([[center[0] + radius*math.cos(angle), center[1] + radius*math.sin(angle)] for angle in angles])


def contour_length(contour):
    return np.sum(np.linalg.norm(contour[1:] - contour[:-1], axis=1))

def interpolate_point_along(contour, dist):
    acc_dist = 0
    for i in range(1, contour.shape[0]):
        acc_dist += np.linalg.norm(contour[i] - contour[i-1])
        if acc_dist >= dist:
            return contour[i]
    return None

def approximate(contour, smoothing):
    length = contour.shape[0]
    spl, u = interpolate.make_splprep(contour.T, s=smoothing*smoothing*length)
    u_new = np.linspace(0, 1, length // 1)
    contour = np.array(spl(u_new)).T
    # Simplify using douglas pecker
    contour = rdp_iter(contour, epsilon=smoothing*0.2) # type: ignore
    return contour

def approximate_all(contours, smoothing):
    return [approximate(contour, smoothing) for contour in contours]


def point_in_polygon(point: NDArray, polygon: NDArray) -> bool:
    return points_in_poly([point], polygon)[0]

