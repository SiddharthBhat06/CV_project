#!/usr/bin/env python3
"""AI Pothole Detection System.

Main application entry point supporting Webcam, Image, and Video detection modes.

Two ways to use it:

  Terminal (CLI / interactive menu):
      python main.py
      python main.py --source video --input road.mp4

  Streamlit web frontend (same file):
      streamlit run main.py
      python main.py --ui          # shortcut that launches the line above
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Streamlit / pandas are only needed for the web UI; the CLI works without them.
try:
    import pandas as pd
    import streamlit as st
except ImportError:  # pragma: no cover
    pd = None
    st = None

from config import config
from detector import (
    PotholeDetector,
    ThreadedInferencePipeline,
)
from utils.civil_metrics import InspectionRegistry, calculate_rhi
from utils.ui_composer import (
    CriticalPotholeTracker,
    combine_views,
    create_thumbnail_sidebar,
)
from utils.video_utils import (
    create_video_writer,
    print_banner,
    print_final_summary,
    print_progress_bar,
    print_webcam_status,
    save_annotated_image,
    save_detection_log,
)


def parse_arguments() -> argparse.Namespace:
    """Parse terminal command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Terminal-Based AI Pothole Detection System (YOLO + OpenCV)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["webcam", "image", "video"],
        help="Input source mode: 'webcam', 'image', or 'video'",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input image or video file (required for 'image' and 'video' modes)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to custom YOLO .pt model weights (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Detection confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--cam-idx",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help="Camera device index for webcam mode (default: 0)",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Run without displaying OpenCV window (useful for headless / automated runs)",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Append a JSONL detection event log to outputs/detection_log.jsonl",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=config.DEFAULT_SKIP_FRAMES,
        metavar="N",
        help=f"Run inference every Nth frame (default: {config.DEFAULT_SKIP_FRAMES})",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit web frontend instead (same as: streamlit run main.py)",
    )
    return parser.parse_args()



# =========================================================================
# Mode Handlers
# =========================================================================

def run_image_mode(
    detector: PotholeDetector,
    image_path_str: str,
    no_view: bool = False,
    save_log: bool = False,
) -> None:

    """Detect potholes in a single image, print stats, and save output."""
    image_path = Path(image_path_str)
    if not image_path.is_file():
        print(f"\n[ERROR] Input image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print_banner(
        source_name=f"Image ({image_path.name})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Could not decode image: {image_path}", file=sys.stderr)
        sys.exit(1)

    result = detector.detect(frame)
    annotated_frame = detector.draw_annotations(frame, result, show_hud=True)
    saved_path = save_annotated_image(annotated_frame, image_path.name)

    if save_log:
        save_detection_log(result, source_name=image_path.name, frame_idx=1)

    log_file = (config.OUTPUTS_DIR / "detection_log.jsonl") if save_log else None

    print_final_summary(
        source_name=image_path.name,
        processed_frames=1,
        total_frames=1,
        unique_potholes=result.count,
        total_instances=result.count,
        highest_confidence=result.max_confidence,
        avg_fps=result.fps,
        output_path=saved_path,
        log_path=log_file,
    )

    if not no_view:
        window_title = f"{config.WINDOW_TITLE} - {image_path.name}"
        ih, iw = annotated_frame.shape[:2]
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, iw, ih)
        cv2.imshow(window_title, annotated_frame)
        print("\nDisplaying image preview. Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def run_video_mode(
    detector: PotholeDetector,
    video_path_str: str,
    no_view: bool = False,
    save_log: bool = False,
    skip_frames: int = 0,
) -> None:

    """Process road video frame-by-frame, display detection, and save annotated video.

    Args:
        detector: Initialised PotholeDetector.
        video_path_str: Path to the input video file.
        no_view: Suppress the OpenCV preview window.
        save_log: Append detection events to outputs/detection_log.jsonl.
        skip_frames: Run inference every Nth frame; reuse last result otherwise.
            0 means every frame (no skipping).
    """
    video_path = Path(video_path_str)
    if not video_path.is_file():
        print(f"\n[ERROR] Input video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = 15.0
    target_frame_time = 1.0 / target_fps
    output_fps = target_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_filename = f"annotated_{video_path.name}"
    writer, out_path = create_video_writer(out_filename, output_fps, (width, height))

    print_banner(
        source_name=f"Video ({video_path.name})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )
    skip_info = f" | Frame-skip: {skip_frames}" if skip_frames > 0 else ""
    print(f"Resolution: {width}x{height} | Frames: {total_frames} | Max 15 FPS | Recursive Loop | ByteTrack{skip_info}")
    print(f"Output: {out_path}\n")

    frame_idx = 0
    total_potholes_found = 0
    fps_history = []
    registry = InspectionRegistry(max_missing_frames=15, sidebar_capacity=4)
    critical_tracker = CriticalPotholeTracker(hysteresis_margin=0.10)
    detector.tracking_mode = True   # enable ByteTrack persistent IDs
    last_result = None

    annotated_frame = None

    if not no_view:
        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        # Window sized for composite (main + 25% sidebar)
        cv2.resizeWindow(config.WINDOW_TITLE, min(1600, width + width // 4), height)

    try:
        with ThreadedInferencePipeline(detector) as pipe:
            while True:
                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Recursive continuous loop until stopped by user
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                frame_idx += 1

                # Frame-skip: feed only every Nth frame to the inference thread;
                # the pipeline automatically returns the last result for skipped frames.
                if skip_frames == 0 or frame_idx % (skip_frames + 1) == 1:
                    pipe.put(frame)

                result = pipe.get(timeout=0.05)
                if result is None:
                    # No result yet (very first frame) — run synchronously as fallback
                    result = detector.detect(frame)

                last_result = result

                # Update critical pothole tracker + inspection registry + RHI
                critical_id = critical_tracker.update(result.detections)
                registry.update_frame(result.detections, frame)
                rhi = calculate_rhi(result.detections)

                total_potholes_found += result.count

                work_duration = time.perf_counter() - loop_start
                remaining_time = max(0.0, target_frame_time - work_duration)
                effective_frame_time = work_duration + remaining_time
                current_fps = min(15.0, 1.0 / max(0.0001, effective_frame_time))
                fps_history.append(current_fps)

                annotated_frame = detector.draw_annotations(
                    frame,
                    result,
                    show_hud=True,
                    total_count=registry.total_unique_count,
                    override_fps=current_fps,
                    critical_id=critical_id,
                    rhi=rhi,
                    total_logged=registry.unique_logged_count,
                )
                # Save clean annotated frame at original resolution (no sidebar baked in)
                writer.write(annotated_frame)

                if save_log and result.count > 0:
                    save_detection_log(result, source_name=video_path.name, frame_idx=frame_idx)

                print_progress_bar(
                    frame_idx=frame_idx % max(1, total_frames),
                    total_frames=total_frames,
                    fps=current_fps,
                    current_potholes=registry.total_unique_count,
                    highest_confidence=result.max_confidence,
                )

                if not no_view:
                    # Build 75/25 composite: main view + inspection log sidebar
                    _fh, _fw = annotated_frame.shape[:2]
                    _sb_w = max(200, _fw // 4)   # sidebar = 25% of main width
                    _sidebar = create_thumbnail_sidebar(
                        registry.crops_queue, _sb_w, _fh)
                    _composite = combine_views(annotated_frame, _sidebar)
                    cv2.imshow(config.WINDOW_TITLE, _composite)
                    wait_ms = max(1, int(remaining_time * 1000)) if remaining_time > 0 else 1
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("\n[INFO] Stopped by user (Q pressed).")
                        break
                else:
                    if remaining_time > 0:
                        time.sleep(remaining_time)

    finally:
        cap.release()
        writer.release()
        if not no_view:
            cv2.destroyAllWindows()

    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
    log_file = (config.OUTPUTS_DIR / "detection_log.jsonl") if save_log else None
    print_final_summary(
        source_name=video_path.name,
        processed_frames=frame_idx,
        total_frames=total_frames,
        unique_potholes=registry.total_unique_count,
        total_instances=total_potholes_found,
        highest_confidence=registry.max_confidence,
        avg_fps=avg_fps,
        output_path=out_path,
        log_path=log_file,
    )


def run_webcam_mode(
    detector: PotholeDetector,
    cam_idx: int = 0,
    no_view: bool = False,
    save_log: bool = False,
    skip_frames: int = 0,
) -> None:

    """Capture live webcam frames, run real-time detection, and display OpenCV window.

    Args:
        detector: Initialised PotholeDetector.
        cam_idx: Camera device index.
        no_view: Suppress OpenCV preview window (headless mode).
        save_log: Append detection events to outputs/detection_log.jsonl.
        skip_frames: Run inference every Nth frame; reuse last result otherwise.
    """
    print_banner(
        source_name=f"Webcam (Device {cam_idx})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )
    print(f"Initializing camera device {cam_idx}...")

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not access webcam at index {cam_idx}.", file=sys.stderr)
        print("Check that a camera is connected and permissions are granted.", file=sys.stderr)
        sys.exit(1)

    print("Camera active. Press 'Q' inside the preview window to exit.\n")

    target_fps = float(config.TARGET_VIDEO_FPS)
    target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.080

    frame_idx = 0
    total_potholes_found = 0
    fps_history = []
    registry = InspectionRegistry(max_missing_frames=15, sidebar_capacity=4)
    critical_tracker = CriticalPotholeTracker(hysteresis_margin=0.10)
    detector.tracking_mode = True   # enable ByteTrack persistent IDs
    last_result = None
    annotated_frame = None

    if not no_view:
        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        # Set initial window size from camera; falls back to 1280x720 if unavailable
        _cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        _cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cv2.resizeWindow(config.WINDOW_TITLE, _cam_w, _cam_h)

    try:
        with ThreadedInferencePipeline(detector) as pipe:
            while True:
                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Failed to read frame from webcam. Retrying...", file=sys.stderr)
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                h, w = frame.shape[:2]

                # Frame-skip: feed only every Nth frame to the inference thread
                if skip_frames == 0 or frame_idx % (skip_frames + 1) == 1:
                    pipe.put(frame)

                result = pipe.get(timeout=0.05)
                if result is None:
                    result = detector.detect(frame)

                last_result = result

                # Update critical pothole tracker + inspection registry + RHI
                critical_id = critical_tracker.update(result.detections)
                registry.update_frame(result.detections, frame)
                rhi = calculate_rhi(result.detections)

                total_potholes_found += result.count

                work_duration = time.perf_counter() - loop_start
                remaining_time = max(0.0, target_frame_time - work_duration)
                effective_frame_time = work_duration + remaining_time
                current_fps = min(15.0, 1.0 / max(0.0001, effective_frame_time))
                fps_history.append(current_fps)

                annotated_frame = detector.draw_annotations(
                    frame,
                    result,
                    show_hud=True,
                    total_count=registry.total_unique_count,
                    override_fps=current_fps,
                    critical_id=critical_id,
                    rhi=rhi,
                    total_logged=registry.unique_logged_count,
                )

                if save_log and result.count > 0:
                    save_detection_log(result, source_name=f"webcam:{cam_idx}", frame_idx=frame_idx)

                print_webcam_status(
                    frame_idx=frame_idx,
                    fps=current_fps,
                    current_count=result.count,
                    total_unique=registry.total_unique_count,
                    highest_confidence=result.max_confidence,
                )

                if not no_view:
                    # Build 75/25 composite: main view + inspection log sidebar
                    _fh, _fw = annotated_frame.shape[:2]
                    _sb_w = max(200, _fw // 4)
                    _sidebar = create_thumbnail_sidebar(
                        registry.crops_queue, _sb_w, _fh)
                    _composite = combine_views(annotated_frame, _sidebar)
                    cv2.imshow(config.WINDOW_TITLE, _composite)
                    wait_ms = max(1, int(remaining_time * 1000)) if remaining_time > 0 else 1
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("\n[INFO] Stopped by user (Q pressed).")
                        break
                else:
                    if remaining_time > 0:
                        time.sleep(remaining_time)

    finally:
        cap.release()
        if not no_view:
            cv2.destroyAllWindows()

    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
    log_file = (config.OUTPUTS_DIR / "detection_log.jsonl") if save_log else None
    print_final_summary(
        source_name=f"Webcam ({cam_idx})",
        processed_frames=frame_idx,
        total_frames=frame_idx,
        unique_potholes=registry.total_unique_count,
        total_instances=total_potholes_found,
        highest_confidence=registry.max_confidence,
        avg_fps=avg_fps,
        log_path=log_file,
    )



# =========================================================================
# Streamlit Web Frontend  (run with:  streamlit run main.py)
# =========================================================================

_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp", "tiff"]
_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v"]
_PREVIEW_MAX_WIDTH = 960


def launch_streamlit_ui() -> None:
    """Start the Streamlit frontend for this file in a subprocess, then exit."""
    if st is None:
        print(
            "\n[ERROR] Streamlit is not installed. Install it with:\n"
            "    pip install streamlit pandas",
            file=sys.stderr,
        )
        sys.exit(1)
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
    print(f"\n[INFO] Launching web UI: {' '.join(cmd)}\n")
    try:
        code = subprocess.call(cmd)
    except KeyboardInterrupt:
        code = 0
    sys.exit(code)


if st is not None:

    @st.cache_resource(show_spinner="Loading YOLO model…")
    def _ui_load_detector(model_path: str, _version: float) -> PotholeDetector:
        """Load the detector once per weights file (``_version`` = file mtime)."""
        return PotholeDetector(model_path=model_path)


def _ui_sidebar() -> dict:
    """Render sidebar controls and return the chosen settings."""
    sb = st.sidebar
    sb.title("⚙️ Settings")
    sb.caption(f"Compute device: **{config.get_device_name()}**")

    sb.subheader("Model")
    model_source = sb.radio(
        "Weights source",
        ["Path on disk", "Upload .pt file"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if model_source == "Path on disk":
        model_path = sb.text_input("Model path", value=str(config.MODEL_PATH))
    else:
        weights = sb.file_uploader("YOLO weights (.pt)", type=["pt"])
        model_path = str(config.MODEL_PATH)
        if weights is not None:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            target = config.MODELS_DIR / f"uploaded_{weights.name}"
            if not target.is_file() or target.stat().st_size != weights.size:
                target.write_bytes(weights.getvalue())
            model_path = str(target)
            sb.caption(f"Saved to `{target.name}`")

    sb.subheader("Detection")
    conf = sb.slider("Confidence threshold", 0.05, 0.95, float(config.CONFIDENCE_THRESHOLD), 0.05)
    iou = sb.slider("NMS IoU threshold", 0.10, 0.90, float(config.IOU_THRESHOLD), 0.05)
    skip_frames = sb.slider(
        "Skip frames (video / webcam)", 0, 5, int(config.DEFAULT_SKIP_FRAMES),
        help="Run inference every Nth frame and reuse the last result in between. 0 = every frame.",
    )
    preview_every = sb.slider(
        "Preview refresh (every N frames)", 1, 10, 2,
        help="Higher values process faster because the browser is updated less often.",
    )
    show_crops = sb.checkbox("Show inspection-log thumbnails in live view", value=True)
    save_log = sb.checkbox("Append events to outputs/detection_log.jsonl", value=False)

    return dict(
        model_path=model_path, conf=conf, iou=iou, skip_frames=skip_frames,
        preview_every=preview_every, show_crops=show_crops, save_log=save_log,
    )


def _ui_reset_tracker(detector: PotholeDetector) -> None:
    """Drop persisted ByteTrack state so a new stream starts with fresh IDs."""
    try:
        predictor = detector.model.predictor
        if predictor is not None and hasattr(predictor, "trackers"):
            del predictor.trackers
    except Exception:
        pass


def _ui_detections_df(detections):
    rows = []
    for i, d in enumerate(detections, start=1):
        rows.append({
            "#": d.track_id if d.track_id is not None else i,
            "Tier": d.tier,
            "Severity": d.severity,
            "Confidence": round(d.confidence, 3),
            "Depth (cm)": round(d.depth_cm, 1),
            "Area (m²)": round(d.area_real_m2, 3),
            "Volume (m³)": round(d.volume_m3, 4),
            "Patch weight (kg)": round(d.weight_kg, 1),
            "BBox (x1,y1,x2,y2)": f"{d.x1},{d.y1},{d.x2},{d.y2}",
        })
    return pd.DataFrame(rows)


def _ui_rgb_preview(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert to RGB and downscale for fast browser display."""
    h, w = frame_bgr.shape[:2]
    if w > _PREVIEW_MAX_WIDTH:
        frame_bgr = cv2.resize(frame_bgr, (_PREVIEW_MAX_WIDTH, int(h * _PREVIEW_MAX_WIDTH / w)))
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _ui_to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to browser-playable H.264 if ffmpeg is installed."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
            check=True, timeout=900,
        )
        return dst.is_file()
    except Exception:
        return False


# ---- image processing (upload + browser snapshot) -----------------------

def _ui_process_image(state_key: str, frame: np.ndarray, name: str,
                      detector: PotholeDetector, settings: dict) -> None:
    """Run detection on one image and store the outcome in session state."""
    _ui_reset_tracker(detector)
    detector.tracking_mode = False
    detector.confidence_threshold = settings["conf"]
    detector.iou_threshold = settings["iou"]

    result = detector.detect(frame)
    rhi = calculate_rhi(result.detections)
    annotated = detector.draw_annotations(frame, result, show_hud=True, rhi=rhi)

    if settings["save_log"]:
        save_detection_log(result, source_name=name, frame_idx=1)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUTS_DIR / f"annotated_{Path(name).stem}.png"
    cv2.imwrite(str(out_path), annotated)
    ok, png = cv2.imencode(".png", annotated)

    st.session_state[state_key] = dict(
        name=name,
        rgb=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        table=_ui_detections_df(result.detections),
        count=result.count,
        max_conf=result.max_confidence,
        inference_ms=result.inference_time_ms,
        rhi=rhi,
        png=png.tobytes() if ok else None,
        saved=str(out_path),
    )


def _ui_show_image_result(state_key: str) -> None:
    res = st.session_state.get(state_key)
    if not res:
        return

    st.image(res["rgb"], caption=f"Annotated — {res['name']}", use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Potholes detected", res["count"])
    c2.metric("Highest confidence", f"{res['max_conf'] * 100:.1f}%")
    c3.metric("Road Health Index", res["rhi"] if res["rhi"] is not None else "—")
    c4.metric("Inference time", f"{res['inference_ms']:.0f} ms")

    if res["count"] > 0:
        st.dataframe(res["table"], use_container_width=True, hide_index=True)
    else:
        st.info("No potholes detected in this image.")

    d1, d2 = st.columns(2)
    if res["png"]:
        d1.download_button(
            "⬇️ Download annotated image", res["png"],
            file_name=f"annotated_{Path(res['name']).stem}.png", mime="image/png",
            key=f"png_{state_key}",
        )
    if res["count"] > 0:
        d2.download_button(
            "⬇️ Download detections (CSV)", res["table"].to_csv(index=False),
            file_name=f"detections_{Path(res['name']).stem}.csv", mime="text/csv",
            key=f"csv_{state_key}",
        )
    st.caption(f"Saved copy: `{res['saved']}`")


# ---- video / webcam streaming -------------------------------------------

class _UIStreamSession:
    """Per-run state shared by the video and webcam processing loops."""

    def __init__(self, detector: PotholeDetector, settings: dict, source_name: str):
        _ui_reset_tracker(detector)
        detector.tracking_mode = True
        detector.confidence_threshold = settings["conf"]
        detector.iou_threshold = settings["iou"]

        self.detector = detector
        self.settings = settings
        self.source_name = source_name

        self.registry = InspectionRegistry(max_missing_frames=15, sidebar_capacity=4)
        self.critical = CriticalPotholeTracker(hysteresis_margin=0.10)

        self.frame_idx = 0
        self.instances = 0
        self.fps = 0.0
        self.fps_history: list = []
        self.last_result = None
        self.tracks: dict = {}

    def process(self, frame: np.ndarray):
        """Detect (honouring frame-skip), track and annotate one frame."""
        t0 = time.perf_counter()
        self.frame_idx += 1
        skip = self.settings["skip_frames"]

        fresh = (
            self.last_result is None
            or skip == 0
            or self.frame_idx % (skip + 1) == 1
        )
        if fresh:
            self.last_result = self.detector.detect(frame)
        result = self.last_result

        critical_id = self.critical.update(result.detections)
        self.registry.update_frame(result.detections, frame)
        rhi = calculate_rhi(result.detections)

        if fresh:
            self.instances += result.count
            self._record_tracks(result)
            if self.settings["save_log"] and result.count > 0:
                save_detection_log(result, source_name=self.source_name, frame_idx=self.frame_idx)

        annotated = self.detector.draw_annotations(
            frame,
            result,
            show_hud=True,
            total_count=self.registry.total_unique_count,
            override_fps=self.fps,
            critical_id=critical_id,
            rhi=rhi,
            total_logged=self.registry.unique_logged_count,
        )

        inst_fps = 1.0 / max(1e-4, time.perf_counter() - t0)
        self.fps = inst_fps if self.fps == 0.0 else 0.1 * inst_fps + 0.9 * self.fps
        self.fps_history.append(inst_fps)
        return annotated, result, rhi

    def _record_tracks(self, result) -> None:
        for d in result.detections:
            if d.track_id is None:
                continue
            rec = self.tracks.setdefault(d.track_id, {
                "Track ID": d.track_id, "Tier": d.tier, "Peak confidence": 0.0,
                "Depth (cm)": 0.0, "Area (m²)": 0.0, "Volume (m³)": 0.0,
                "Patch weight (kg)": 0.0, "Frames seen": 0,
            })
            rec["Frames seen"] += 1
            rec["Peak confidence"] = max(rec["Peak confidence"], round(d.confidence, 3))
            rec["Depth (cm)"] = max(rec["Depth (cm)"], round(d.depth_cm, 1))
            rec["Area (m²)"] = max(rec["Area (m²)"], round(d.area_real_m2, 3))
            rec["Volume (m³)"] = max(rec["Volume (m³)"], round(d.volume_m3, 4))
            rec["Patch weight (kg)"] = max(rec["Patch weight (kg)"], round(d.weight_kg, 1))
            rec["Tier"] = d.tier

    def preview(self, annotated: np.ndarray) -> np.ndarray:
        """Live preview, optionally composited with the inspection-log sidebar."""
        if self.settings["show_crops"]:
            try:
                h, w = annotated.shape[:2]
                side = create_thumbnail_sidebar(self.registry.crops_queue, max(200, w // 4), h)
                annotated = combine_views(annotated, side)
            except Exception:
                pass
        return _ui_rgb_preview(annotated)

    def render_metrics(self, placeholder, result, rhi) -> None:
        with placeholder.container():
            c = st.columns(5)
            c[0].metric("Unique potholes", self.registry.total_unique_count)
            c[1].metric("Active now", result.count)
            c[2].metric("Logged", self.registry.unique_logged_count)
            c[3].metric("Road Health Index", rhi if rhi is not None else "—")
            c[4].metric("FPS", f"{self.fps:.1f}")

    def summary(self, **extra) -> dict:
        table = pd.DataFrame(sorted(self.tracks.values(), key=lambda r: r["Track ID"]))
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0
        return dict(
            source=self.source_name,
            frames=self.frame_idx,
            unique=self.registry.total_unique_count,
            instances=self.instances,
            max_conf=self.registry.max_confidence,
            avg_fps=avg_fps,
            table=table,
            log_path=str(config.OUTPUTS_DIR / "detection_log.jsonl") if self.settings["save_log"] else None,
            **extra,
        )


def _ui_show_summary(state_key: str) -> None:
    """Display the stored summary of a finished (or stopped) run."""
    s = st.session_state.get(state_key)
    if not s:
        return

    st.subheader(f"Run summary — {s['source']}")
    c = st.columns(5)
    c[0].metric("Frames processed", s["frames"])
    c[1].metric("Unique potholes", s["unique"])
    c[2].metric("Total detections", s["instances"])
    c[3].metric("Highest confidence", f"{s['max_conf'] * 100:.1f}%")
    c[4].metric("Avg FPS", f"{s['avg_fps']:.1f}")

    out = s.get("video_path")
    if out and Path(out).is_file():
        data = Path(out).read_bytes()
        if s.get("playable"):
            st.video(data)
        else:
            st.info(
                "The annotated video is saved with the `mp4v` codec, which most browsers can't "
                "play inline. Download it below (install `ffmpeg` for an in-browser preview)."
            )
        st.download_button(
            "⬇️ Download annotated video", data,
            file_name=Path(out).name, mime="video/mp4", key=f"vid_{state_key}",
        )
        st.caption(f"Saved copy: `{out}`")

    table = s["table"]
    if not table.empty:
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download pothole table (CSV)", table.to_csv(index=False),
            file_name="potholes_summary.csv", mime="text/csv", key=f"csv_{state_key}",
        )
    if s.get("log_path"):
        st.caption(f"Event log: `{s['log_path']}`")


def _ui_run_video(uploaded, detector: PotholeDetector, settings: dict) -> None:
    """Process an uploaded video frame by frame with a live preview."""
    suffix = Path(uploaded.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        os.unlink(tmp.name)
        st.error(f"Could not open video: {uploaded.name}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps <= 0:
        src_fps = float(config.TARGET_VIDEO_FPS)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(uploaded.name).stem
    raw_path = config.OUTPUTS_DIR / f"annotated_{stem}_raw.mp4"
    final_path = config.OUTPUTS_DIR / f"annotated_{stem}.mp4"
    writer = cv2.VideoWriter(
        str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height)
    )

    session = _UIStreamSession(detector, settings, uploaded.name)
    st.session_state.pop("video_summary", None)

    metrics_ph = st.empty()
    frame_ph = st.empty()
    progress = st.progress(0.0)
    status = st.empty()
    status.caption("Processing… press **⏹ Stop** to end early (output so far is kept).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            annotated, result, rhi = session.process(frame)
            writer.write(annotated)

            if session.frame_idx % settings["preview_every"] == 0:
                frame_ph.image(session.preview(annotated), channels="RGB", use_container_width=True)
                session.render_metrics(metrics_ph, result, rhi)
                if total:
                    progress.progress(min(1.0, session.frame_idx / total))
    finally:
        # Runs on normal end, on Stop (Streamlit rerun) and on errors. No st.* calls here.
        cap.release()
        writer.release()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

        playable = _ui_to_h264(raw_path, final_path)
        if playable:
            raw_path.unlink(missing_ok=True)
            out = final_path
        else:
            out = raw_path
        st.session_state["video_summary"] = session.summary(
            video_path=str(out), playable=playable
        )

    progress.progress(1.0)
    status.success("Processing complete.")


def _ui_run_webcam(cam_idx: int, detector: PotholeDetector, settings: dict) -> None:
    """Live detection from a camera attached to the machine running Streamlit."""
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)
    if not cap.isOpened():
        st.error(
            f"Could not open camera {cam_idx}. Check that a camera is connected, "
            "permissions are granted, and try another index."
        )
        return

    session = _UIStreamSession(detector, settings, f"webcam:{cam_idx}")
    st.session_state.pop("webcam_summary", None)

    metrics_ph = st.empty()
    frame_ph = st.empty()
    st.caption("Live — switch **Start camera** off to stop.")

    failures = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                failures += 1
                if failures > 30:
                    st.error("Camera stopped delivering frames.")
                    break
                time.sleep(0.05)
                continue
            failures = 0

            annotated, result, rhi = session.process(frame)
            if session.frame_idx % settings["preview_every"] == 0:
                frame_ph.image(session.preview(annotated), channels="RGB", use_container_width=True)
                session.render_metrics(metrics_ph, result, rhi)
    finally:
        cap.release()
        st.session_state["webcam_summary"] = session.summary()


# ---- tabs ---------------------------------------------------------------

def _ui_tab_image(detector, settings) -> None:
    st.subheader("Detect potholes in an image")
    uploaded = st.file_uploader("Upload a road image", type=_IMAGE_TYPES, key="img_upload")
    if uploaded is not None:
        left, right = st.columns([1, 3])
        left.image(uploaded, caption="Original", use_container_width=True)
        if right.button("🔍 Run detection", type="primary", key="img_run"):
            frame = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                st.error("Could not decode that image.")
            else:
                with st.spinner("Detecting…"):
                    _ui_process_image("image_result", frame, uploaded.name, detector, settings)
    _ui_show_image_result("image_result")


def _ui_tab_video(detector, settings) -> None:
    st.subheader("Detect potholes in a video")
    uploaded = st.file_uploader("Upload a road video", type=_VIDEO_TYPES, key="vid_upload")
    c1, c2, _ = st.columns([1, 1, 4])
    start = c1.button("▶️ Start", type="primary", disabled=uploaded is None, key="vid_start")
    c2.button("⏹ Stop", key="vid_stop", help="Ends processing early and keeps the output so far.")

    if start and uploaded is not None:
        _ui_run_video(uploaded, detector, settings)
    _ui_show_summary("video_summary")


def _ui_tab_camera(detector, settings) -> None:
    st.subheader("Camera")
    mode = st.radio(
        "Mode", ["Live (camera on this machine)", "Browser snapshot"],
        horizontal=True, key="cam_mode",
    )

    if mode.startswith("Live"):
        st.caption(
            "Reads a camera attached to the computer **running Streamlit**. "
            "If the app is hosted remotely, use *Browser snapshot* instead."
        )
        cam_idx = st.number_input(
            "Camera device index", min_value=0, max_value=10,
            value=int(config.DEFAULT_CAMERA_INDEX), step=1,
        )
        if st.toggle("Start camera", key="cam_run"):
            _ui_run_webcam(int(cam_idx), detector, settings)
        _ui_show_summary("webcam_summary")
    else:
        photo = st.camera_input("Take a picture of the road")
        if photo is not None:
            data = photo.getvalue()
            sig = hash(data)
            if st.session_state.get("snap_sig") != sig:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with st.spinner("Detecting…"):
                        _ui_process_image("snapshot_result", frame, "browser_snapshot.png",
                                          detector, settings)
                st.session_state["snap_sig"] = sig
        _ui_show_image_result("snapshot_result")


def run_streamlit_app() -> None:
    """Entry point for the Streamlit frontend."""
    st.set_page_config(page_title="AI Pothole Detection", page_icon="🕳️", layout="wide")
    st.title("🕳️ AI Pothole Detection System")
    st.caption("YOLO + ByteTrack detection with civil-engineering metrics.")

    settings = _ui_sidebar()

    model_file = Path(settings["model_path"])
    # The default path can be absent on first launch: PotholeDetector downloads
    # PeterHdd's pretrained weights automatically. Custom paths must already exist.
    is_default_model = model_file.resolve() == config.DEFAULT_MODEL_PATH.resolve()
    if not model_file.is_file() and not is_default_model:
        st.error(
            f"Model weights not found at `{model_file}`. Choose an existing `.pt` file "
            f"or restore the default path `{config.DEFAULT_MODEL_PATH}` to use the "
            "PeterHdd pretrained model."
        )
        st.stop()
    try:
        model_version = model_file.stat().st_mtime if model_file.is_file() else 0
        detector = _ui_load_detector(str(model_file), model_version)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to initialise detector: {exc}")
        st.stop()

    tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Image", "🎞️ Video", "📷 Camera"])
    with tab_img:
        _ui_tab_image(detector, settings)
    with tab_vid:
        _ui_tab_video(detector, settings)
    with tab_cam:
        _ui_tab_camera(detector, settings)


# =========================================================================
# Main Execution Entrypoint
# =========================================================================

def _pick_file_dialog(file_type: str = "video") -> str | None:
    """Open a native file picker dialog and return selected path (or None)."""
    try:
        import os
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()          # hide the blank root window
        root.attributes("-topmost", True)
        if file_type == "image":
            file_path = filedialog.askopenfilename(
                title="Select an Image File",
                initialdir=os.path.expanduser("~"),
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                    ("All files", "*.*"),
                ],
            )
        else:
            file_path = filedialog.askopenfilename(
                title="Select a Video File",
                initialdir=os.path.expanduser("~"),
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                    ("All files", "*.*"),
                ],
            )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"  [WARN] Native file dialog unavailable ({e}). Enter path manually.")
        return None


def interactive_menu() -> argparse.Namespace:
    """Display interactive terminal menu and return populated Namespace."""
    print()
    print("=" * 56)
    print("      AI POTHOLE DETECTION SYSTEM (TERMINAL & LIVE CV)")
    print("=" * 56)
    print()
    print("  Select detection mode:")
    print()
    print("  [1]  Live Webcam — Real-time camera detection in OpenCV")
    print("  [2]  Video File  — Browse & detect with live OpenCV tracking")
    print("  [3]  Web UI      — Launch the Streamlit frontend in your browser")
    print()
    print("=" * 56)

    while True:
        choice = input("  Enter choice (1, 2 or 3): ").strip()
        if choice in ("1", "2"):
            break
        if choice == "3":
            launch_streamlit_ui()
        print("  Invalid choice. Enter 1, 2 or 3.")

    ns = argparse.Namespace(
        model=None,
        conf=config.CONFIDENCE_THRESHOLD,
        cam_idx=config.DEFAULT_CAMERA_INDEX,
        no_view=False,
        source=None,
        input=None,
        save_log=False,
        skip_frames=config.DEFAULT_SKIP_FRAMES,
    )

    if choice == "1":
        ns.source = "webcam"
        cam_input = input("\n  Camera device index [default: 0]: ").strip()
        ns.cam_idx = int(cam_input) if cam_input.isdigit() else 0
    elif choice == "2":
        ns.source = "video"
        print("\n  Opening file browser... (select your video file)")
        picked = _pick_file_dialog(file_type="video")
        if picked:
            print(f"  Selected: {picked}")
            ns.input = picked
        else:
            # Fallback: manual entry
            while True:
                path_str = input("  Enter path to video file (or drag & drop): ").strip().strip('"').strip("'")
                if Path(path_str).is_file():
                    ns.input = path_str
                    break
                print(f"  [ERROR] File not found: {path_str}. Try again.")

    return ns


def main():
    # `--ui` must be handled before argparse because --source is otherwise required
    if "--ui" in sys.argv[1:]:
        launch_streamlit_ui()

    # If args were passed use the original argparse flow; otherwise show menu
    if len(sys.argv) > 1:
        args = parse_arguments()
    else:
        args = interactive_menu()

    # Input validation
    if args.source in ("image", "video") and not args.input:
        print(
            f"\n[ERROR] The --input argument is required when using '--source {args.source}'.\n"
            f"Example: python main.py --source {args.source} --input road.{'jpg' if args.source == 'image' else 'mp4'}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize YOLO Pothole Detector
    try:
        detector = PotholeDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
        )
    except Exception as e:
        print(f"\n[INITIALIZATION ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.source == "image":
        run_image_mode(
            detector=detector,
            image_path_str=args.input,
            no_view=args.no_view,
            save_log=args.save_log,
        )
    elif args.source == "video":
        run_video_mode(
            detector=detector,
            video_path_str=args.input,
            no_view=args.no_view,
            save_log=args.save_log,
            skip_frames=args.skip_frames,
        )
    elif args.source == "webcam":
        run_webcam_mode(
            detector=detector,
            cam_idx=args.cam_idx,
            no_view=args.no_view,
            save_log=args.save_log,
            skip_frames=args.skip_frames,
        )



def _running_in_streamlit() -> bool:
    """True when this script is being executed by ``streamlit run``."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# Under `streamlit run main.py` __name__ is also "__main__", so check Streamlit first.
if _running_in_streamlit():
    run_streamlit_app()
elif __name__ == "__main__":
    main()
