from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw
import asyncio

# Colors for different layout elements
CATEGORY_COLORS = {
    'title': (255, 0, 0),  # Red
    'text': (0, 255, 0),  # Green
    'list': (255, 165, 0),  # Orange
    'table': (0, 255, 255),  # Cyan
    'figure': (0, 0, 255),  # Blue
    'header': (255, 0, 255),  # Magenta
    'footer': (128, 0, 128),  # Purple
    'reference': (165, 42, 42)  # Brown
}

# Default color for unknown categories
DEFAULT_COLOR = (128, 128, 128)  # Gray


async def draw_layout_boxes(
        image: Image.Image,
        layout_results: List[Dict[str, Any]],
        line_width: int = 2
) -> Image.Image:
    """
    Draw bounding boxes on the image for detected layout elements.

    Args:
        image: Source image
        layout_results: List of detected layout elements with their properties
        line_width: Width of the bounding box lines (default: 2)

    Returns:
        Image with drawn bounding boxes
    """
    result_image = await asyncio.to_thread(lambda: image.copy())
    draw = ImageDraw.Draw(result_image)

    async def process_blocks():
        for block in layout_results:
            box = await _get_box_coordinates(block['bbox'])
            color = await _get_category_color(block['type'])
            await _draw_box(draw, box, color, line_width)

    await process_blocks()
    return result_image


async def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box coordinates [x1, y1, x2, y2]
        box2: Second box coordinates [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    box1_arr = np.array(box1)
    box2_arr = np.array(box2)

    x_left = max(box1_arr[0], box2_arr[0])
    y_top = max(box1_arr[1], box2_arr[1])
    x_right = min(box1_arr[2], box2_arr[2])
    y_bottom = min(box1_arr[3], box2_arr[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1_arr[2] - box1_arr[0]) * (box1_arr[3] - box1_arr[1])
    box2_area = (box2_arr[2] - box2_arr[0]) * (box2_arr[3] - box2_arr[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


async def _get_box_coordinates(bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    Extract box coordinates from bbox.

    Args:
        bbox: Bounding box coordinates

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates
    """
    return (
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3]
    )


async def _get_category_color(category: str) -> Tuple[int, int, int]:
    """
    Get RGB color for a given category.

    Args:
        category: Layout element category

    Returns:
        RGB color tuple
    """
    category = category.lower()
    return CATEGORY_COLORS.get(category, DEFAULT_COLOR)


async def _draw_box(
        draw: ImageDraw.ImageDraw,
        box: Tuple[float, float, float, float],
        color: Tuple[int, int, int],
        line_width: int
) -> None:
    """
    Draw a single bounding box on the image.

    Args:
        draw: ImageDraw object
        box: Box coordinates (x1, y1, x2, y2)
        color: RGB color tuple
        line_width: Width of the box lines
    """
    await asyncio.to_thread(
        draw.rectangle,
        [
            (box[0], box[1]),
            (box[2], box[3])
        ],
        outline=color,
        width=line_width
    )
