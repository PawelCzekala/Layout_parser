# Layout Detection API

**FastAPI-based microservice** for detecting and evaluating document layout elements (titles, text, tables, figures, etc.) using **PaddleOCR** and **PPStructure**.

<img width="400" height="532" alt="layout_detection" src="https://github.com/user-attachments/assets/1ca10261-55cf-4f45-b49f-f83f857a20e4" />

---

## Features

- **/detect/** — Detect layout elements and return annotated image.
- **/evaluate/** — Compare layout detection with ground-truth JSON and return metrics + visual output.
- Containerized via **Docker**.
- Interactive UI via **Swagger** (`/docs`).

---

## Requirements

- Python 3.9
- Docker (for containerized deployment)
- `make` 

---

## API Endpoints

### `GET /`

Health check endpoint.

**Response:**
```json
{ "status": "OK" }
```

### `POST /detect/`
Detect layout in uploaded image.

**Input**: image file (.jpg, .png, etc.)

**Output**: annotated image (PNG) with bounding boxes

**Swagger usage**:

Click `/detect/`, select image as form-data, execute.

Response: raw image (shown or downloadable)

### `POST /evaluate/`
Evaluate predicted layout against ground-truth COCO-style JSON.

**Input**:

img_file: image file (.jpg, .png, etc.)

json_gt: ground-truth .json file

**Output**:

JSON with:

precision, recall, f1_score, iou_mean, tp, fp, fn

image: base64-encoded annotated image

**Swagger usage**:

Upload both image and .json

Returns metrics and encoded image

Example response:

```json
{
  "precision": 0.87,
  "recall": 0.85,
  "f1_score": 0.86,
  "iou_mean": 0.71,
  "tp": 18,
  "fp": 3,
  "fn": 2,
  "image": "data:image/png;base64,iVBORw0K..."
}
```

---

## Deployment with Docker
1. Build image
```bash
make build
```
2. Run container
```bash
make run
```
Service will be available at http://localhost:8000

3. Open Swagger UI

Go to: http://localhost:8000/docs

---

## Project Structure
```bash
layout_parser/
├── app/
│   ├── main.py          # FastAPI app
│   ├── detection.py     # Detection logic using PPStructure
│   ├── evaluation.py    # Evaluation logic and metrics
│   └── utils.py         # Drawing, IOU, etc.
├── requirements.txt
├── Dockerfile
├── Makefile
└── README.md
```

---

## Ground-Truth Format
Expected .json format (COCO-style):

```json
{
  "categories": [{ "id": 1, "name": "title" }],
  "annotations": [
    { "bbox": ["x", "y", "width", "height"],
      "category_id": 1 },
  ]
}
```

---

## Notes
Supported categories: title, text, list, table, figure, etc.

IoU threshold = 0.5 for TP/FP/FN classification.

Color-coded evaluation output:

✅ TP: green

❌ FP: red

🟦 FN: blue

---

## License
MIT License. Use at your own risk.
