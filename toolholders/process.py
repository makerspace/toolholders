from dataclasses import dataclass
from enum import Enum
from typing import Callable
from joblib import Memory
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt
import skimage.morphology as morphology
from skimage.measure import regionprops, find_contours
from scipy.spatial import ConvexHull, QhullError
import numpy as np
from PIL import Image
from numpy.typing import NDArray
import skimage.transform
import matplotlib.pyplot as plt
import math

from toolholders.bbox import bbox_area, bbox_union
from toolholders.geometry import approximate_all, contour_length, interpolate_point_along
memory = Memory("cachedir")

class LabelPosition(Enum):
    Left = "left"
    Right = "right"
    Top = "top"
    Bottom = "bottom"
    Auto = "auto"

def polyArea(points: np.ndarray) -> float:
    return 0.5*np.abs(np.dot(points[:,0],np.roll(points[:,1],1))-np.dot(points[:,1],np.roll(points[:,0],1)))

@dataclass
class GridConfig:
    stride: tuple[int,int]
    chessboard: bool

def find_grid_mounting_holes(valid_position_mask: NDArray, hole_edge_clearance_mm: float, hole_hole_clearance_mm: float, mm2pixels: float, grid: GridConfig, debug: bool=False) -> tuple[list[np.ndarray], float]:
    skeleton_mask = morphology.skeletonize(valid_position_mask)
    distances = distance_transform_edt(~skeleton_mask, return_distances=True)
    valid_position_mask = morphology.isotropic_erosion(valid_position_mask, mm2pixels*hole_edge_clearance_mm)

    # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    if debug:
        fig, ax = plt.subplots()
        ax.imshow(distances, cmap=plt.cm.gray)
    # ax2.imshow(distances, cmap=plt.cm.gray)

    def sample_mask(mask, pos_px):
        if pos_px[0] < 0 or pos_px[0] >= mask.shape[0] or pos_px[1] < 0 or pos_px[1] >= mask.shape[1]:
            return False
        return mask[pos_px[0], pos_px[1]]
    
    def snap_to_pixel(pos):
        r = int(pos[0]*mm2pixels)
        c = int(pos[1]*mm2pixels)
        return np.array([r, c])

    size_mm = np.array(valid_position_mask.shape) / mm2pixels
    grid_size = (size_mm/grid.stride).astype(int) + 2
    # Ensure grid size is odd
    grid_size += (grid_size+1) % 2

    def find_grid_mounting_holes_with_offset(normalized_grid_offset: tuple[float,float], ax) -> tuple[list[np.ndarray], float]:
        offset = (size_mm / 2) - grid.stride * ((grid_size-1)//2)
        offset += normalized_grid_offset * np.array(grid.stride) * 2

        # Draw grid
        grid_points = []
        grid_colors = []
        pixels2mm = 1 / mm2pixels

        def mirror_h(r, c):
            return np.array([r, grid_size[1]-1 - c])

        valid_holes = []
        hole_scores = []
        hole_grid_coordinates = []
        for r in range(0, grid_size[0]):
            for c in range(0, grid_size[1]):
                if grid.chessboard and (r % 2) == (c % 2):
                    # Grid has a chessboard pattern
                    continue

                p = np.array([r,c]) * grid.stride + offset
                k = snap_to_pixel(p)
                grid_points.append(p)
                if sample_mask(valid_position_mask, k):
                    grid_colors.append('green')
                    
                    k = snap_to_pixel(p)
                    distance = distances[k[0], k[1]] * pixels2mm # type: ignore
                    
                    distance_score = -0.5 * float(distance)**2
                    valid_holes.append(p)
                    hole_scores.append(distance_score)
                    hole_grid_coordinates.append((r, c))
                else:
                    grid_colors.append('red')
        
        def symmetry_score(hole1: int, hole2: int):
            if abs(normalized_grid_offset[1]) > 0.05:
                return 0
            
            # Check distance between them
            if np.linalg.norm(valid_holes[hole1][0] - valid_holes[hole2][1]) < hole_hole_clearance_mm:
                return -float('inf')

            # Check if holes are symmetric
            (r1, c1) = hole_grid_coordinates[hole1]
            (r2, c2) = hole_grid_coordinates[hole2]
            if np.all(mirror_h(r1, c1) == (r2, c2)):
                return 30
            return 0

        best_hole_indices, best_score = optimize_hole_subset(valid_holes, hole_scores, [40, 100, 0, -60], pixels2mm, symmetry_score)
        best_holes = [valid_holes[idx] for idx in best_hole_indices]

        if ax is not None:
            ax.scatter([p[1]*mm2pixels for p in grid_points], [p[0]*mm2pixels for p in grid_points], color=grid_colors)
            # ax2.scatter([p[1]*mm2pixels for p in grid_points], [p[0]*mm2pixels for p in grid_points], color=grid_colors)
            # ax3.scatter([p[1]*mm2pixels for p in grid_points], [p[0]*mm2pixels for p in grid_points], color=grid_colors)
            ax.scatter([p[1]*mm2pixels for p in best_holes], [p[0]*mm2pixels for p in best_holes], color='blue')
            ax.set_aspect('equal')
            # Set title to score
            ax.set_title(f"Score: {best_score}")

        return best_holes, best_score

    best_mounting_holes = []
    best_mounting_holes_score = -float('inf')
    offsets = np.linspace(0, 1, 5, endpoint=False)
    if debug:
        fig, axes = plt.subplots(offsets.shape[0],offsets.shape[0])
    else:
        axes = None
    for i, offsetr in enumerate(offsets):
        for j, offsetc in enumerate(offsets):
            mounting_holes, score = find_grid_mounting_holes_with_offset((offsetr, offsetc), axes[i,j] if axes is not None else None)
            if score > best_mounting_holes_score and len(mounting_holes) > 0:
                best_mounting_holes_score = score
                best_mounting_holes = mounting_holes
    
    return best_mounting_holes, best_mounting_holes_score

def optimize_hole_subset(holes: list[NDArray], intrinsic_scores: list[float], hole_count_scores: list[float], pixels2mm: float, combination_score: Callable[[int,int], float]):
    best_score = -float('inf')
    best_holes: list[int] = []
    best_convex_hull = 0
    intrinsic_scores_arr = np.array(intrinsic_scores)

    combination_cache = np.zeros((len(holes), len(holes)), dtype=float)
    for i in range(len(holes)):
        for j in range(len(holes)):
            combination_cache[i, j] = combination_score(i, j)

    # Iterate over all subsets of holes
    for subset_mask in range(1, 1 << len(holes)):
        subset_indices = [idx for idx in range(len(holes)) if (subset_mask & (1 << idx)) != 0]

        score = 0 # unit: square millimeters
        convex_hull_area = 0
        area_score = 0
        # Convex hull
        if len(subset_indices) >= 3:
            subset_holes = [holes[i] for i in subset_indices]
            try:
                hull = ConvexHull(subset_holes)
                convex_hull_vertices = np.array([subset_holes[i] for i in hull.vertices])
                convex_hull_area = polyArea(convex_hull_vertices) * pixels2mm * pixels2mm
            except QhullError:
                # Likely all points are collinear
                convex_hull_area = 0
            area_score += 1.0 * math.sqrt(convex_hull_area)

        # First two holes are good, but if we get more than 3 holes, we get a penalty
        intrinsic_score = np.sum(intrinsic_scores_arr[subset_indices])
        count_score = 0
        symmetry_score = 0

        for subset_hole_index, hole_index in enumerate(subset_indices):
            count_score += hole_count_scores[min(subset_hole_index, len(hole_count_scores)-1)]
            symmetry_score += np.sum(combination_cache[hole_index, subset_indices])
        
        score += intrinsic_score
        score += area_score
        score += count_score
        score += symmetry_score

        if score > best_score:
            print(f"I: {intrinsic_score} A: {area_score} C: {count_score} S: {symmetry_score} => {score}")
            best_score = score
            best_holes = subset_indices
            best_convex_hull = convex_hull_area
        
    return best_holes, float(best_score)

def hole_positions(skeleton: list[np.typing.NDArray], distance: float, max_hole_distance: float, mm2pixels: float, max_holes: int | None = None) -> list[np.ndarray]:
    points = []
    for contour in skeleton:
        total_length = contour_length(contour)
        mid_length = total_length / 2
        if mm2pixels*distance < mid_length:
            start = interpolate_point_along(contour, mm2pixels*distance)
            end = interpolate_point_along(np.flip(contour, axis=0), mm2pixels*distance)
            points.append(start)
            points.append(end)
        
        # if start is None or total_length > mm2pixels*max_hole_distance:
        mid = interpolate_point_along(contour, mid_length)
        assert mid is not None
        points.append(mid)
    points_filtered = [p for p in points if p is not None]

    def distance_score(i: int, j: int) -> float:
        if i == j:
            return 0
        dist = np.linalg.norm(points_filtered[i] - points_filtered[j])
        if dist < mm2pixels*distance:
            return -50
        if dist < mm2pixels*max_hole_distance:
            return -10
        return 0

    count_scores = [40.0, 100, -30, -100]
    if max_holes is not None:
        while len(count_scores) < max_holes+1:
            count_scores.append(count_scores[-1])
        count_scores = count_scores[:max_holes+1]
        count_scores[-1] = -float('inf')

    hole_indices, best_score = optimize_hole_subset(points_filtered, [0] * len(points_filtered), count_scores, mm2pixels, distance_score)
    best_holes = [points_filtered[idx] for idx in hole_indices]
    return best_holes

def place_label(valid_position_mask, current_shape_mask, label: str, label_size: tuple[int,int], label_position: LabelPosition):
        assert valid_position_mask.shape == current_shape_mask.shape

        # bbox
        current_shape_bbox = regionprops(current_shape_mask.astype(int))[0].bbox

        def place_left(label_size: tuple[int,int], flip_lr: bool, transpose: bool, symmetry_axis: int) -> tuple[tuple[int, int], float, NDArray]:
            pos_mask = valid_position_mask
            shape_mask = current_shape_mask
            if transpose:
                pos_mask = np.transpose(pos_mask)
                shape_mask = np.transpose(shape_mask)
                label_size = (label_size[1], label_size[0])
            if flip_lr:
                pos_mask = np.fliplr(pos_mask)
                shape_mask = np.fliplr(shape_mask)

            height = label_size[0]
            leftmost = np.zeros((pos_mask.shape[0],), dtype=np.int32) + pos_mask.shape[1]
            leftmost_shape = np.zeros((pos_mask.shape[0],), dtype=np.int32) + pos_mask.shape[1]
            valid = np.zeros((pos_mask.shape[0],))
            for r in range(pos_mask.shape[0]):
                for c in range(pos_mask.shape[1]):
                    if pos_mask[r,c]:
                        leftmost[r] = c
                        valid[r] += 1
                        break
                
                for c in range(pos_mask.shape[1]):
                    if shape_mask[r,c]:
                        leftmost_shape[r] = c
                        valid[r] += 1
                        break
            
            rolling = []
            coords = []
            for r in range(0, pos_mask.shape[0]-height):
                x = np.min(leftmost[r:r+height])
                bbox = (r, int(x) - label_size[1], r+height, int(x))
                extra_bbox_area = bbox_area(bbox_union(bbox, current_shape_bbox)) - bbox_area(current_shape_bbox)
                extra_area = np.sum(leftmost_shape[r:r+height] - x)
                # If bbox overlaps with vertical centerline, give it a bonus
                symmetric1 = bbox[0] < pos_mask.shape[0] / 2 and bbox[2] > pos_mask.shape[0] / 2
                symmetric2 = bbox[1] < pos_mask.shape[1] / 2 and bbox[3] > pos_mask.shape[1] / 2
                exactly_symmetric1 = (bbox[0]+bbox[2])//2 == pos_mask.shape[0] // 2
                exactly_symmetric2 = (bbox[1]+bbox[3])//2 == pos_mask.shape[1] // 2
                score = (
                    2*extra_area
                    + extra_bbox_area
                    - 1 * (label_size[0] * label_size[1] if symmetric1 and symmetry_axis == 0 else 0)
                    - 1 * (label_size[0] * label_size[1] if symmetric2 and symmetry_axis == 1 else 0)
                    - 0.5 * (label_size[0] * label_size[1] if exactly_symmetric1 and symmetry_axis == 0 else 0)
                    - 0.5 * (label_size[0] * label_size[1] if exactly_symmetric2 and symmetry_axis == 1 else 0)
                )

                if np.sum(valid[r:r+height] >= 2) < height*0.6:
                    score = float('inf')

                coords.append(int(x))
                rolling.append(score)
            
            if len(rolling) == 0:
                return (0, 0), float('inf'), np.zeros_like(pos_mask, dtype=np.bool)

            i = np.argmin(rolling)
            p = (int(i), coords[i] - label_size[1])
            result_score = float(rolling[i])

            fill_shape = np.zeros_like(pos_mask, dtype=np.bool)
            for r in range(p[0], p[0]+label_size[0]):
                if valid[r] == 2:
                    fill_shape[r, max(p[1]+label_size[1],0):leftmost_shape[r]] = True

            if flip_lr:
                p = (p[0], p[1] + label_size[1])
                p = (p[0], pos_mask.shape[1] - p[1] - 1)
                fill_shape = np.fliplr(fill_shape)
            if transpose:
                p = (p[1], p[0])
                label_size = (label_size[1], label_size[0])
                fill_shape = np.transpose(fill_shape)
            
            return p, result_score, fill_shape

        p1, s1, f1 = place_left(label_size, flip_lr=False, transpose=False, symmetry_axis=1)
        p2, s2, f2 = place_left(label_size, flip_lr=True, transpose=False, symmetry_axis=1)
        p3, s3, f3 = place_left(label_size, flip_lr=False, transpose=True, symmetry_axis=0)
        p4, s4, f4 = place_left(label_size, flip_lr=True, transpose=True, symmetry_axis=0)

        positions = [p1, p2, p3, p4]
        fills = [f1, f2, f3, f4]
        print(positions)
        print("Label size", label_size)
        scores = [s1, s2, s3, s4]
        print("Scores", scores)
        best = np.argmin(scores)

        # sizes = [label_size, label_size, label_size, label_size]
        # colors = ["red", "blue", "green", "purple"]
        # fig, ax = plt.subplots()
        # ax.imshow(valid_position_mask)
        # outline_contours = approximate_all(find_contours(current_shape_mask), 0.3)
        # for contour in outline_contours:
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

        # # Create rectangles
        # for pos, size, color, fill, score in zip(positions, sizes, colors, fills, scores):
        #     ax.add_patch(Rectangle((pos[1], pos[0]), size[1], size[0], edgecolor=color, facecolor='none'))
        #     # Text at label
        #     ax.text(pos[1], pos[0], str(score), fontsize=10, color='black')
        #     ax.imshow(fill, alpha=1.0*fill, cmap="jet")
        
        # ax.set_aspect('equal')
        # plt.show()

        label_bbox = (positions[best][0], positions[best][1], positions[best][0] + label_size[0], positions[best][1] + label_size[1])
        return label_bbox, fills[best]

@memory.cache
def find_symmetric_axis(mask):
    flipped = np.fliplr(mask)
    best_score = 0
    best_theta = 0
    for theta in range(-30, 30, 5):
        rotated_mask = skimage.transform.rotate(flipped, theta)
        overlap = np.sum(mask & rotated_mask)
        union = np.sum(mask | rotated_mask)
        score = overlap / union
        if score > best_score:
            best_score = score
            best_theta = theta
    
    for theta in np.arange(best_theta - 5, best_theta + 5, 0.5):
        rotated_mask = skimage.transform.rotate(flipped, theta)
        overlap = np.sum(mask & rotated_mask)
        union = np.sum(mask | rotated_mask)
        score = overlap / union
        if score > best_score:
            best_score = score
            best_theta = theta
    
    if best_theta > 180:
        best_theta -= 360
    
    return best_theta/2

def segment_image(image_path: str, max_size: int) -> tuple[NDArray, NDArray]:
    from transformers import pipeline

    img = Image.open(image_path)
    # Resize to NxN, keeping aspect ratio
    img.thumbnail((max_size, max_size))

    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device="cpu")
    pillow_mask = pipe(img, return_mask = True) # outputs a pillow mask
    pillow_image = np.array(pipe(img)) # applies mask on input and returns a pillow image

    # Convert pillow mask to numpy array
    mask = np.array(pillow_mask)

    return pillow_image, mask

def roll_centroid_to_center(mask):
    centroid = np.array(regionprops(mask.astype(int))[0].centroid)
    # Ensure centroid is in the middle of the image by padding it
    center = np.array(mask.shape) / 2
    diff = center - centroid
    def f(img):
        assert mask.shape[0:2] == img.shape[0:2], f"{mask.shape[0:2]} != {img.shape[0:2]}"
        return np.roll(img, diff.astype(int), axis=(0, 1))
    return f