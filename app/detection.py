from typing import List, Dict, Any
import io
import numpy as np
from PIL import Image
from paddleocr import PPStructure
import asyncio

from app.utils import draw_layout_boxes

CATEGORY_COLORS = {
    'title': 'red',
    'text': 'green',
    'list': 'orange',
    'table': 'cyan',
    'figure': 'blue'
}


async def detect_layout_with_visualization(image_bytes: bytes) -> Image.Image:
    """
    Detect layout in the provided image and return visualization.

    Args:
        image_bytes: Raw image bytes

    Returns:
        PIL Image with detected layout visualization
    """
    image = await asyncio.to_thread(Image.open, io.BytesIO(image_bytes))
    image = await asyncio.to_thread(lambda x: x.convert("RGB"), image)
    np_image = np.array(image)

    structure = PPStructure(layout=True, show_log=False)
    result = await asyncio.to_thread(structure, np_image)

    result_image = await draw_layout_boxes(image, result)
    return result_image


async def classify_boxes(boxes: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Classify detected boxes into different layout elements.
    
    Args:
        boxes: List of box coordinates
    
    Returns:
        List of dictionaries containing bbox coordinates and labels
    """
    async def process_boxes():
        labeled_boxes = []
        for box in boxes:
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            
            height = y_max - y_min
            label = 'title' if height > 30 else 'text'
            
            labeled_boxes.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "label": label
            })
        return labeled_boxes

    return await asyncio.to_thread(process_boxes)
