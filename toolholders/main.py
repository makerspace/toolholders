from dataclasses import dataclass, replace
from enum import Enum
import time
from typing import Literal

from matplotlib.patches import Rectangle

from toolholders.bbox import bbox_area, bbox_center, bbox_grow, bbox_relative_to, bbox_union
t0 = time.time()
from PIL import Image, ImageFont
import os.path
from dataclasses_json import DataClassJsonMixin
import numpy as np
from skimage import filters, morphology
from skimage.segmentation import flood_fill
from skimage.measure import regionprops, find_contours, grid_points_in_poly
import skimage.transform
import math
import matplotlib.pyplot as plt
from matplotlib import colormaps
import dijkstra3d
import trace_skeleton
import argparse
import os
from numpy.typing import NDArray
import matplotlib

from toolholders.export import export_dxf
from toolholders.geometry import approximate_all, circle, point_in_polygon
from toolholders.morphology import crop_to_bbox, crop_to_fit_bbox, dilate_direction, triangle_footprint
from toolholders.pack import move_packed_shapes, pack_shapes, rotate_shape_to_minimize_bbox
from toolholders.process import GridConfig, LabelPosition, find_grid_mounting_holes, find_symmetric_axis, hole_positions, place_label, roll_centroid_to_center, segment_image
from toolholders.shapes import AffineTransform, Group, Shape, ShapeType, Text
t1 = time.time()

# Use tk
# matplotlib.use('tkagg')

print(f"Import took {t1-t0:.2f}")

def leftmost_object(mask) -> NDArray:
    # Find leftmost true pixel
    coord = None
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[j, i]:
                coord = (j, i)
                break
        if coord is not None:
            break
    
    mask = mask.astype(int)
    mask = flood_fill(mask, coord, 2)
    return mask == 2

def calculate_mm2pixels(mask, real_size_mm: tuple[float, float]):
    props = regionprops(mask.astype(int))
    mm2pixels = math.sqrt(props[0].area / (real_size_mm[0]*real_size_mm[1]))
    return mm2pixels

@dataclass
class Config:
    packing_margin: float
    ruler_size_mm: tuple[float, float]
    hole_through_diameter_mm: float
    hole_thread_diameter_mm: float
    hole_resolution: int
    mounting_hole_clearance_mm: float
    mounting_hole_grid_size_mm: float
    mounting_hole_min_distance_mm: float
    assembly_hole_margin_from_edge_mm: float
    assembly_hole_max_distance_mm: float
    border_width_mm: float
    lift_height_mm: float
    max_support_angle: float
    cover_thickness_mm: float
    contour_smoothing: float
    material_thickness: float
    high_quality: bool

@dataclass
class DebugConfig:
    show_contours: bool = False

class GridType(Enum):
    IkeaSkadis = "ikea_skadis"
    ElfaClassic = "elfa_classic"

@dataclass
class ToolholderSettings(DataClassJsonMixin):
    active_contours: list[int] | None = None
    label: str | None = None
    label_size: tuple[float,float] | Literal["auto"] | None = None
    label_position: LabelPosition | None = None
    overrides: dict[str, str | float | int | bool] | None = None
    thickness: float | None = 10.0
    url: str | None = None
    grid: GridType = GridType.IkeaSkadis

def segment_image_cached(image_path: str, high_quality: bool):
    t0 = time.time()
    suffix = "_hq" if high_quality else ""
    cache_path = "cache/" + os.path.splitext(os.path.basename(image_path))[0]
    masked_image_cache = cache_path + f"_masked{suffix}.png"
    mask_cache = cache_path + f"_mask{suffix}.png"
    if os.path.exists(masked_image_cache) and os.path.exists(mask_cache):
        result = np.array(Image.open(masked_image_cache)), np.array(Image.open(mask_cache))
        print(f"Loaded cached image in {time.time()-t0:.2f}")
        return result
    
    masked_image, mask = segment_image(image_path, 2048 if high_quality else 1024)
    Image.fromarray(masked_image).save(masked_image_cache)
    Image.fromarray(mask).save(mask_cache)
    return masked_image, mask
    
def process_image(image_path: str, config: Config, toolholder: ToolholderSettings, debug: DebugConfig) -> ToolholderSettings:
    if toolholder.overrides is not None:
        config = replace(config, **toolholder.overrides)

    masked_image, grayscale_mask = segment_image_cached(image_path, high_quality=config.high_quality)
    mask = grayscale_mask > 128
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.remove_small_objects(mask, min_size=64)
    mask = morphology.remove_small_holes(mask, area_threshold=64)
    

    # Check dimensions of ruler
    ruler = leftmost_object(mask)
    mm2pixels = calculate_mm2pixels(ruler, config.ruler_size_mm)
    pixels2mm = 1 / mm2pixels
    print(f"Scale: {mm2pixels:.2} pixels/mm")

    def millimeters(mm):
        return int(mm * mm2pixels)

    # Remove ruler from mask
    mask = mask & ~ruler
    grayscale_mask[ruler] = 0

    cropper = crop_to_bbox(mask, millimeters(30))
    mask = cropper(mask)
    grayscale_mask = cropper(grayscale_mask)
    masked_image = cropper(masked_image)
    roller = roll_centroid_to_center(mask)
    mask = roller(mask)
    grayscale_mask = roller(grayscale_mask)
    masked_image = roller(masked_image)
    theta = find_symmetric_axis(mask)
    print(f"Symmetric axis: {theta}")

    # Orient the shape according to its moments
    # props = regionprops(mask.astype(np.uint8))
    # angle = props[0].orientation
    # mask = skimage.transform.rotate(mask, -theta)
    grayscale_mask = np.array(Image.fromarray(grayscale_mask).rotate(-float(theta), resample=Image.Resampling.BICUBIC, expand=False))
    masked_image = skimage.transform.rotate(masked_image, -theta)

    # We rotate the grayscale mask, as it interpolates a lot better than a binary mask
    # Then we do all the noise reduction again
    threshold = 0.25
    blur_radius = 1
    mask = filters.gaussian(grayscale_mask, sigma=blur_radius)
    mask = mask > threshold
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.remove_small_objects(mask, min_size=64)
    mask = morphology.remove_small_holes(mask, area_threshold=64)

    cropper = crop_to_bbox(mask, millimeters(30))
    mask = cropper(mask)
    masked_image = cropper(masked_image)

    def support_edges(mask, max_support_angle: float):
        intensity = mask.astype(float)
        intensity = filters.gaussian(intensity, sigma=millimeters(2))
        intensity = intensity / intensity.max()
        v_edges = filters.sobel_v(intensity)
        h_edges = filters.sobel_h(intensity)
        edges = np.maximum(0.0001, np.hypot(v_edges, h_edges))
        h_edges /= edges
        v_edges /= edges

        # Filter edges which slope downwards
        top_supports = h_edges < -math.cos(math.radians(max_support_angle))
        return top_supports
    
    top_supports = support_edges(mask, config.max_support_angle)
    field = ~morphology.binary_erosion(mask, morphology.disk(2))
    outer_edge = (field & mask)

    masked_image_nd = np.array(masked_image)
    red_mask = (masked_image_nd[:, :, 0] > 0.9)  & (masked_image_nd[:, :, 1] < 0.2) & (masked_image_nd[:, :, 2] < 0.2)
    red_mask = morphology.binary_dilation(red_mask, morphology.disk(3))
    top_edge = top_supports & outer_edge & ~red_mask
    top_edge = morphology.binary_closing(top_edge, morphology.disk(3))

    # Remove small edges. Multiply size by 2 since the edges are typically 2 pixels wide
    top_edge = morphology.remove_small_objects(top_edge, min_size=millimeters(11)*2)

    lift_mask = morphology.binary_dilation(mask, np.flipud(triangle_footprint(millimeters(config.lift_height_mm), math.pi/12)))

    outline_mask = morphology.isotropic_dilation(mask, millimeters(config.border_width_mm))
    outline_mask = ~morphology.flood(outline_mask, (0,0))

    def prompt_keep_regions(top_edge, labels) -> list[int]:
        fig, ax = plt.subplots()
        ax.imshow(masked_image)
        ax.imshow(top_edge, cmap=colormaps["jet"], alpha=1.0 * top_edge)
        regions = regionprops(labels)
        for i, region in enumerate(regions):
            # Centroid
            y0, x0 = region.centroid
            ax.annotate(f"{region.label}",
                    xy=(x0, y0), xycoords='data',
                    xytext=(1.5, 1.5), textcoords='offset points', color='red')

        plt.show(block=False)

        # Prompt for which to keep
        while True:
            nums = input("Which supports should be used? (space separated numbers)\n").strip().split()
            try:
                kept_regions = [int(num) for num in nums]
                if any(num > len(regions) or num < 1 for num in kept_regions):
                    print(f"Invalid region. Should be between 1 and {len(regions)}")
                    continue
            except ValueError:
                print("Invalid number. Please enter space separated numbers.")
                continue
            break
    
        # Hide plot
        plt.close()
        return kept_regions
    
    # f, ax = plt.subplots()
    # ax.imshow(masked_image)
    # ax.imshow(top_edge, cmap=plt.cm.jet, alpha=1.0 * top_edge)
    # ax.imshow(red_mask, cmap=plt.cm.jet, alpha=1.0 * red_mask)
    # plt.show()
    labels = morphology.label(top_edge)
    if toolholder.active_contours is None:
        toolholder.active_contours = prompt_keep_regions(top_edge, labels)
    
    top_edge = np.zeros_like(top_edge)
    for num in toolholder.active_contours:
        top_edge |= labels == num
    
    t3 = time.time()
    edge_coordinates = np.argwhere(top_edge)

    # Add dummy dimension to field to make it 3d
    field = field[:, :, np.newaxis]
    # Add dummy dimension to edge_coordinates to make it 3d
    edge_coordinates2 = [(coord[0], coord[1], 0) for coord in edge_coordinates]

    assert len(edge_coordinates2) > 0
    distances = dijkstra3d.euclidean_distance_field(field, source=edge_coordinates2, anisotropy=(1,1,1))[:,:,0]
    thickness = millimeters(config.border_width_mm + 2)
    np.nan_to_num(distances, False, thickness, thickness, thickness)
    for i in range(8):
        distances[mask] = thickness
        distances = filters.gaussian(distances, sigma=millimeters(2))

    distances = np.minimum(distances, thickness)
    support_mask_smooth = distances < thickness
    support_mask_smooth &= ~lift_mask
    support_mask_smooth &= outline_mask

    t0 = time.time()
    print(f"Support mask took {t0-t3:.2f}")
    for i in range(10):
        prev = support_mask_smooth
        support_mask_smooth = morphology.isotropic_opening(support_mask_smooth, millimeters(3))
        support_mask_smooth &= ~mask
        if np.all(support_mask_smooth == prev):
            break
    t1 = time.time()
    print(f"Opening took {t1-t0:.2f}")

    cover_mask = dilate_direction(support_mask_smooth, "up", millimeters(config.cover_thickness_mm + 2))
    cover_range = morphology.isotropic_dilation(mask, millimeters(config.cover_thickness_mm))
    for i in range(10):
        prev = cover_mask
        cover_mask = morphology.isotropic_opening(cover_mask, millimeters(7))
        cover_mask &= cover_range
        cover_mask |= support_mask_smooth
        if np.all(cover_mask == prev):
            break
        prev = cover_mask
    t2 = time.time()
    print(f"Cover mask took {t2-t1:.2f}")

    if toolholder.label_size is None:
        toolholder.label_size = "auto"
    
    label_font = ImageFont.truetype("./BebasNeue-Regular.ttf", 20)
    tiny_font = ImageFont.truetype("./BebasNeue-Regular.ttf", 5)

    if toolholder.label is None:
        f, ax = plt.subplots()
        ax.imshow(masked_image)
        plt.show(block=False)
        toolholder.label = input("Enter label for toolholder:\n")
        plt.close(f)

    if toolholder.label_size == "auto":
        bbox = label_font.getbbox(toolholder.label)
        label_size = (bbox[3] - bbox[1], bbox[2] - bbox[0])
        padding = 5
        label_size = (label_size[0] + 2*padding, label_size[1] + 2*padding)
    else:
        label_size = toolholder.label_size


    label_bbox, label_fill = place_label(mask | support_mask_smooth, outline_mask, toolholder.label, (millimeters(label_size[0]), millimeters(label_size[1])), LabelPosition.Left)
    outline_mask |= label_fill
    new_bbox = bbox_union(bbox_grow(label_bbox, 10), (0, 0, mask.shape[0], mask.shape[1]))
    
    mask = crop_to_fit_bbox(mask, new_bbox)
    masked_image = crop_to_fit_bbox(masked_image, new_bbox)
    support_mask_smooth = crop_to_fit_bbox(support_mask_smooth, new_bbox)
    outline_mask = crop_to_fit_bbox(outline_mask, new_bbox)
    label_fill = crop_to_fit_bbox(label_fill, new_bbox)
    cover_mask = crop_to_fit_bbox(cover_mask, new_bbox)
    label_bbox = bbox_relative_to(label_bbox, new_bbox)
    print("Label bbox", label_bbox)

    label_mask = np.zeros_like(mask)
    label_mask[label_bbox[0]:label_bbox[2], label_bbox[1]:label_bbox[3]] = True
    outline_mask |= label_mask
    outline_mask = ~morphology.flood(outline_mask, (0,0))

    outline_contours = approximate_all(find_contours(outline_mask), config.contour_smoothing)
    support_contours = approximate_all(find_contours(support_mask_smooth), config.contour_smoothing)
    cover_contours = approximate_all(find_contours(cover_mask), config.contour_smoothing)
    # Approximate contours
    approx_contours = approximate_all(find_contours(mask), config.contour_smoothing)

    # Run calculations for each region separately, as they are independent
    suppport_mask_labels: NDArray = morphology.label(support_mask_smooth) # type: ignore
    holes = []
    tiny_label_positions = []
    for i in range(1, suppport_mask_labels.max() + 1):
        support_mask_region = suppport_mask_labels == i
        skeleton = trace_skeleton.from_numpy(support_mask_region)
        # Change x,y to row,col
        skeleton = [np.array([[p[1], p[0]] for p in contour]) for contour in skeleton]
        region_holes = hole_positions(skeleton, config.assembly_hole_margin_from_edge_mm, config.assembly_hole_max_distance_mm, mm2pixels)
        holes.extend(region_holes)
        
        for h in region_holes:
            support_mask_region[int(h[0]), int(h[1])] = False
        support_mask_region = morphology.binary_erosion(support_mask_region, morphology.disk(millimeters(3)))
        skeleton2 = trace_skeleton.from_numpy(support_mask_region)
        skeleton2 = [np.array([[p[1], p[0]] for p in contour]) for contour in skeleton2]
        tiny_label_positions.extend(hole_positions(skeleton2, 10000, 10000, mm2pixels, max_holes=1))

    all_shapes = []
    assert len(outline_contours) == 1

    hole_shapes_thread = [Shape(circle(hole * pixels2mm, config.hole_thread_diameter_mm/2, config.hole_resolution), ShapeType.Cut) for hole in holes]

    if toolholder.grid == GridType.IkeaSkadis:
        grid_config = GridConfig((20,20), True)
    elif toolholder.grid == GridType.ElfaClassic:
        grid_config = GridConfig((32,12), False)
    else:
        raise ValueError("Unknown grid type")

    best_mounting_holes, best_mounting_holes_score = find_grid_mounting_holes(outline_mask & ~mask & ~support_mask_smooth & ~cover_mask & ~label_mask, config.mounting_hole_clearance_mm, config.mounting_hole_min_distance_mm, mm2pixels, grid_config, debug=False)

    # plt.show()
    # plt.close(fig)
    if len(best_mounting_holes) == 0:
        print("No mounting holes found")

    mounting_hole_shapes = [Shape(circle(hole, config.hole_through_diameter_mm/2, config.hole_resolution), ShapeType.Cut) for hole in best_mounting_holes]
    text_curves = Text(toolholder.label, label_font, AffineTransform.translate(np.array(bbox_center(label_bbox)) * pixels2mm), ShapeType.EngraveFill, anchor="mm").to_curves(0.01)
    tiny_label_text = toolholder.label[0] + os.path.basename(os.path.splitext(image_path)[0]).split("_")[-1][1:]

    all_shapes.append(Group([
        *[Shape(contour * pixels2mm, ShapeType.Cut) for contour in outline_contours],
        *[Shape(contour * pixels2mm, ShapeType.EngraveOutline) for contour in approx_contours],
        *[Shape(contour * pixels2mm, ShapeType.EngraveOutline) for contour in support_contours],
        *hole_shapes_thread,
        *mounting_hole_shapes,
        text_curves,
        *[
            Text(tiny_label_text, tiny_font, AffineTransform.translate(np.array(label_pos) * pixels2mm), ShapeType.EngraveOutline, anchor="mm").to_curves(0.01)
            for label_pos in tiny_label_positions
        ]
    ]))

    if toolholder.thickness is None:
        while True:
            try:
                toolholder.thickness = float(input("Enter thickness of toolholder [mm]:\n"))
            except ValueError:
                print("Invalid thickness. Please enter a number.")
                continue
            if toolholder.thickness < 0:
                print("Invalid thickness. Please enter a positive number.")
                continue
            break


    support_count = int(max(1, math.ceil(toolholder.thickness / config.material_thickness)))
    for (duplicates, contours) in [(support_count, support_contours), (1, cover_contours)]:
        mirror = AffineTransform.mirror_horizontal()
        for contour in contours:
            shape = Group([
                Shape(contour * pixels2mm, ShapeType.Cut),
                *[Shape(circle(hole * pixels2mm, config.hole_through_diameter_mm/2, config.hole_resolution), ShapeType.Cut) for hole in holes if point_in_polygon(hole, contour)],
                *[
                    # Mirror the text. It will get mirrored again later so that it's actually readable
                    Text(tiny_label_text, tiny_font, mirror, ShapeType.EngraveOutline, anchor="mm")
                    .to_curves(0.01)
                    .with_transform(AffineTransform.translate(np.array(label_pos) * pixels2mm))
                    for label_pos in tiny_label_positions if point_in_polygon(label_pos, contour)
                ]
            ]).with_transform(mirror)
            for _ in range(duplicates):
                all_shapes.append(shape)

    # Draw packed shapes
    packing_margin = 2
    all_shapes = [rotate_shape_to_minimize_bbox(shape) for shape in all_shapes]
    packing_size, rects = pack_shapes(all_shapes, packing_margin)
    packed_shapes = move_packed_shapes(rects, packing_margin)

    export_dxf(packed_shapes, "output/" + os.path.basename(os.path.splitext(image_path)[0]) + "_packed.dxf")

    if False:
        # Remove unstable supports: ones which to not provide any horizontal support
        # E.g. under the tip of scissors
        horizontal_support_overlap = support_mask_smooth & (dilate_direction(mask, "right", 40) | dilate_direction(mask, "left", 40))
        horizontal_support_overlap = morphology.binary_opening(horizontal_support_overlap, morphology.disk(5))
        for i in range(10):
            horizontal_support_overlap = morphology.binary_dilation(horizontal_support_overlap, morphology.disk(5))
            horizontal_support_overlap &= support_mask_smooth

        support_mask_smooth &= horizontal_support_overlap
    
    # Plot contours
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    gray = colormaps["gray"]
    ax1.imshow(top_edge, cmap=gray)
    ax2.imshow(mask, cmap=gray)
    ax3.imshow(mask, cmap=gray)
    ax4.imshow(masked_image, cmap=gray)
    for contour in approx_contours:
        # ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax3.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax4.plot(contour[:, 1], contour[:, 0], linewidth=2, color='black')

    for contour in support_contours:
        ax4.plot(contour[:, 1], contour[:, 0], linewidth=2, color='blue')

    for contour in cover_contours:
        ax4.plot(contour[:, 1], contour[:, 0], linewidth=2, color='purple')

    for contour in skeleton:
        contour = np.array(contour)
        ax4.plot(contour[:, 1], contour[:, 0], linewidth=2, color='green')
    
    for c in text_curves.children:
        ax4.plot(c.contour[:, 1]*mm2pixels, c.contour[:, 0]*mm2pixels, linewidth=2, color='green') # type: ignore

    all_holes = holes + [p*mm2pixels for p in best_mounting_holes]
    ax4.plot([p[1] for p in all_holes], [p[0] for p in all_holes], 'o', color='black')

    assert len(outline_contours) == 1
    ax4.plot(outline_contours[0][:, 1], outline_contours[0][:, 0], linewidth=2, color='red')
    os.makedirs("contours", exist_ok=True)
    fig.savefig("contours/" + os.path.basename(os.path.splitext(image_path)[0]) + "_contours.png")

    if debug.show_contours:
        plt.show()

    plt.close(fig)

    return toolholder

config = Config(
    packing_margin=2,
    ruler_size_mm=(150,70),
    hole_through_diameter_mm=3,
    hole_thread_diameter_mm=2.5,
    hole_resolution=20,
    mounting_hole_clearance_mm=3,
    mounting_hole_grid_size_mm=20,
    mounting_hole_min_distance_mm=15,
    assembly_hole_max_distance_mm=100,
    assembly_hole_margin_from_edge_mm=10,
    border_width_mm=15,
    lift_height_mm=12,
    max_support_angle=75,
    cover_thickness_mm=18,
    contour_smoothing=0.3,
    material_thickness=6,
    high_quality=False,
)

def process(p: str, debug_options: DebugConfig):
    config_path = os.path.splitext(p)[0] + ".json"
    toolholder_config = ToolholderSettings()
    if os.path.exists(config_path):
        with open(config_path) as f:
            try:
                toolholder_config = ToolholderSettings.from_json(f.read())
            except Exception as e:
                print("Failed to load config. Using default")

    toolholder_config = process_image(p, config, toolholder_config, debug_options)

    with open(config_path, "w") as f:
        f.write(toolholder_config.to_json(indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the image to process")
    parser.add_argument("--show", help="Show the result", action="store_true")
    parser.add_argument("--hq", help="Use high quality mode", action="store_true")

    args = parser.parse_args()
    debug_options = DebugConfig(show_contours=args.show)
    image_path = args.image
    if args.hq:
        config = replace(config, high_quality=True)

    if os.path.isdir(image_path):
        for file in os.listdir(image_path):
            if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg"] and "black" not in file:
                p = os.path.join(image_path, file)
                print("Processing", p)
                try:
                    toolholder = process(p, debug_options)
                except Exception as e:
                    print("Failed to process", p)
                    print(e)
    else:
        toolholder = process(image_path, debug_options)