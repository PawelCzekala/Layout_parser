from typing import Dict, Any
import io
import base64
from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import asyncio

from app.detection import detect_layout_with_visualization
from app.evaluation import evaluate_layout

app = FastAPI(title="Layout Detection API")


@app.post("/detect/", response_class=StreamingResponse)
async def detect(img_file: UploadFile = File(...)) -> StreamingResponse:
    """
    Detect layout in the uploaded image.

    Args:
        img_file: Uploaded image file

    Returns:
        StreamingResponse: Processed image with detected layout

    Raises:
        HTTPException: If the uploaded file is empty
    """
    image_bytes = await img_file.read()
    await img_file.close()

    if not image_bytes:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Empty image file"
        )

    result_image = await detect_layout_with_visualization(image_bytes)

    buf = io.BytesIO()
    await asyncio.to_thread(result_image.save, buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/evaluate/", response_class=JSONResponse)
async def evaluate(
        img_file: UploadFile = File(...),
        json_gt: UploadFile = File(...)
) -> JSONResponse:
    """
    Evaluate layout detection against ground truth.

    Args:
        img_file: Image file to evaluate
        json_gt: Ground truth JSON file

    Returns:
        Dict containing evaluation metrics and processed image
    """
    image_bytes, gt_json = await asyncio.gather(
        img_file.read(),
        json_gt.read()
    )

    result_image, metrics = await evaluate_layout(image_bytes, gt_json)

    buf = io.BytesIO()
    await asyncio.to_thread(result_image.save, buf, format="PNG")
    buf.seek(0)

    encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    response_data = dict(metrics)
    response_data["image"] = f"data:image/png;base64,{encoded_img}"

    return JSONResponse(response_data)


@app.get("/")
async def health_check() -> Dict[str, str]:
    """Check API health status."""
    return {"status": "OK"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8100,
        reload=True
    )