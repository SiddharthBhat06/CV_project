# Pothole Detection Models

The application uses PeterHdd's pretrained YOLOv8s pothole detector by default.

## Default model

- Hugging Face: https://huggingface.co/peterhdd/pothole-detection-yolov8
- Downloaded checkpoint: `best.pt`
- Local destination: `models/pothole_yolo.pt`
- Class: `pothole` (single class)

On first launch, if `models/pothole_yolo.pt` does not exist, the detector downloads the checkpoint automatically. An internet connection is required for this first download. The checkpoint is cached locally for later runs.

## Use a custom model instead

You can still use a local `.pt` checkpoint by passing its path:

```bash
python main.py --source image --input road.jpg --model models/my_model.pt
```

Or set `POTHOLE_MODEL_PATH` to a valid local `.pt` file.

## Detection settings

Defaults are tuned to improve recall compared with the previous configuration:

- Confidence threshold: `0.25` (set `POTHOLE_CONFIDENCE_THRESHOLD` to override)
- GPU image size: `832` (set `POTHOLE_GPU_IMG_SIZE` to override)
- CPU image size: `416` (set `POTHOLE_CPU_IMG_SIZE` to override)

Lowering confidence can reveal more detections but can also increase false positives. A larger image size can help with small potholes but may reduce processing speed. Validate these settings against your own road images.
