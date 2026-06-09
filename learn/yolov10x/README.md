
# YOLOv10: Real-Time Fire and Smoke Detection

This repository contains a YOLOv10 model trained for real-time fire and smoke detection. The model uses the Ultralytics YOLO framework to perform object detection with high accuracy and efficiency. Users can adjust the confidence and IoU thresholds for optimal detection results.

## Model Details

- **Model Type**: YOLOv8 (adapted for YOLOv10 features)
- **Task**: Object Detection
- **Framework**: PyTorch
- **Input Size**: Adjustable (default: 640x640)
- **Classes Detected**: Fire, Smoke
- **File**: `best.pt`

## How to Use the Model

This model is hosted on Hugging Face and can be accessed via the **Inference Widget** or programmatically using the Hugging Face Transformers pipeline.

### Inference Widget

Upload an image to the widget below and adjust the following:
- **Confidence Threshold**: Minimum confidence level for predictions (default: 0.25).
- **IoU Threshold**: Minimum IoU level for object matching (default: 0.45).
- **Image Size**: Resize input image (default: 640x640).

### Usage with Python

To use the model programmatically:

```python
import torch
from ultralytics import YOLO
from PIL import Image

model = YOLO('best.pt.bz2')

# Run inference
fn = "https://source.roboflow.com/4yCe2goCmUhzFDmTTBdl0OghOQ02/01sskxb7SHbGiRrHyQek/thumb.jpg"
image = Image.open(fn)
results = model.predict(image, conf=0.25, iou=0.45)
for result in results:
    result.show()