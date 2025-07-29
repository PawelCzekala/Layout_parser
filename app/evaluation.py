from typing import Tuple, List, Dict, Any
import io
import json
import numpy as np
import cv2
from PIL import Image
from paddleocr import PPStructure
import asyncio

from app.utils import calculate_iou

CATEGORY_COLORS_EVAL = {
    'TP': (0, 255, 0),  # Green
    'FP': (0, 0, 255),  # Red
    'FN': (255, 0, 0),  # Blue
}


async def detect_layout(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Detect layout in the provided image.
    """
    image = await asyncio.to_thread(Image.open, io.BytesIO(image_bytes))
    image = await asyncio.to_thread(lambda x: x.convert("RGB"), image)
    np_image = np.array(image)

    structure = PPStructure(layout=True, show_log=False)
    result = await asyncio.to_thread(structure, np_image)

    boxes = []
    for block in result:
        boxes.append({
            'bbox': block['bbox'],
            'label': block['type'].lower()
        })

    return boxes


async def evaluate_layout(
        image_bytes: bytes,
        gt_json_bytes: bytes
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Evaluate layout detection against ground truth.
    """
    image = await asyncio.to_thread(Image.open, io.BytesIO(image_bytes))
    image = await asyncio.to_thread(lambda x: x.convert("RGB"), image)
    np_image = np.array(image).copy()

    pred_boxes = await detect_layout(image_bytes)
    gt_data = json.loads(gt_json_bytes)

    gt_boxes = await _process_ground_truth(gt_data)
    metrics, matches = await _evaluate_predictions(pred_boxes, gt_boxes)

    for box, status in matches:
        x1, y1, x2, y2 = map(int, box)
        await asyncio.to_thread(
            cv2.rectangle,
            np_image,
            (x1, y1),
            (x2, y2),
            CATEGORY_COLORS_EVAL[status],
            2
        )

    result_image = await asyncio.to_thread(Image.fromarray, np_image)
    return result_image, metrics


async def _process_ground_truth(gt_data: Dict) -> List[Dict[str, Any]]:
    """Process ground truth annotations into a standardized format."""
    gt_boxes = []
    cat_map = {cat["id"]: cat["name"].lower()
               for cat in gt_data["categories"]}

    for ann in gt_data["annotations"]:
        x_min, y_min, w, h = ann["bbox"]
        gt_boxes.append({
            "bbox": [x_min, y_min, x_min + w, y_min + h],
            "label": cat_map[ann["category_id"]]
        })

    return gt_boxes



async def _evaluate_predictions(
        pred_boxes: List[Dict[str, Any]],
        gt_boxes: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Tuple]]:
    """
    Evaluate predictions against ground truth.
    """
    tp, fp, fn = 0, 0, 0
    iou_total = 0.0
    matched_gt = set()
    matches = []

    for pred in pred_boxes:
        best_iou = 0
        best_idx = -1

        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt or pred["label"] != gt["label"]:
                continue
            iou = await calculate_iou(pred["bbox"], gt["bbox"])

            if await asyncio.to_thread(lambda: iou > best_iou):
                best_iou = iou
                best_idx = idx

        if best_iou >= 0.5:
            tp += 1
            iou_total += best_iou
            matched_gt.add(best_idx)
            matches.append((pred["bbox"], "TP"))
        else:
            fp += 1
            matches.append((pred["bbox"], "FP"))

    for idx, gt in enumerate(gt_boxes):
        if idx not in matched_gt:
            fn += 1
            matches.append((gt["bbox"], "FN"))

    metrics = await _calculate_metrics(tp, fp, fn, iou_total)

    return metrics, matches


async def _calculate_metrics(
        tp: int,
        fp: int,
        fn: int,
        iou_total: float
) -> Dict[str, Any]:
    """Calculate evaluation metrics."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou_mean = iou_total / tp if tp else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "iou_mean": round(iou_mean, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
