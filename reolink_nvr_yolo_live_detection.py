# Requires: reolink-aio, ultralytics, opencv-python
# Install with: pip install reolink-aio ultralytics opencv-python

import asyncio
import time
from typing import Optional

import cv2
from reolink_aio.api import Host
import torch
from ultralytics import YOLO

# NVR / camera configuration
HOST_IP = "192.168.1.100"
USERNAME = "admin"
PASSWORD = "Fishfish"
RTSP_PORT = 554
CHANNEL = 0  # Camera channel
STREAM_TYPE = "main"  # "main" or "sub"

# YOLO configuration
MODEL_PATH = "C:\\Users\\Leonardo Boran\\Desktop\\MAI\\Code\\cfd-yolov12x-1.00.pt" # Replace with your .pt path if needed
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.45
IMG_SIZE = 640  # Lower this (e.g. 512) for more speed or up to 1024 for more accuracy if your GPU can handle it

# Inference device and precision
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE != "cpu"

# Performance options
FRAME_SKIP = 1  # Process every (FRAME_SKIP + 1)th frame
FRAME_FLUSH_GRABS = 2  # Drop stale buffered frames to reduce lag buildup
WINDOW_NAME = "Reolink YOLO Live Detection"


def open_stream(url: str) -> cv2.VideoCapture:
    """Open RTSP stream with low-latency hints where supported."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_latest_frame(cap: cv2.VideoCapture, flush_grabs: int) -> tuple[bool, Optional[cv2.typing.MatLike]]:
    """Read the most recent frame by discarding older buffered frames."""
    ok = False
    frame = None
    for _ in range(max(0, flush_grabs)):
        if not cap.grab():
            break
    ok, frame = cap.retrieve()
    if not ok or frame is None:
        ok, frame = cap.read()
    return ok, frame


def run_detection_loop(cap: cv2.VideoCapture, model: YOLO) -> None:
    """Read frames, run YOLO, and display annotated detections."""
    frame_idx = 0
    fps_timer = time.time()
    fps_counter = 0
    fps_value = 0.0
    last_annotated = None

    while True:
        ok, frame = read_latest_frame(cap, FRAME_FLUSH_GRABS)
        if not ok or frame is None:
            print("Stream read failed. Attempting to continue...")
            time.sleep(0.1)
            continue

        should_process = (frame_idx % (FRAME_SKIP + 1)) == 0
        if should_process:
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                device=DEVICE,
                half=USE_HALF,
                verbose=False,
            )
            annotated = results[0].plot()
            last_annotated = annotated
        else:
            annotated = last_annotated if last_annotated is not None else frame

        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_value = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()

        cv2.putText(
            annotated,
            f"FPS: {fps_value:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1


async def get_stream_url(host: Host, channel: int, stream_type: str) -> Optional[str]:
    """Connect to host data and fetch RTSP URL for selected channel."""
    print("Connecting to NVR...")
    await host.get_host_data()

    print("Connected successfully")
    print(f"Is NVR: {host.is_nvr}, Number of channels: {host.num_channels}")
    print(f"Available stream channels: {host.stream_channels}")

    if channel not in host.stream_channels:
        print(f"Error: channel {channel} is not available on this device")
        return None

    print(f"Camera name: {host.camera_name(channel)}")

    if "RTSP" not in host.capabilities.get("Host", set()):
        print("Warning: RTSP may not be available on this device")

    print(f"Getting RTSP stream source for channel {channel}...")
    return await host.get_rtsp_stream_source(channel, stream=stream_type)


async def main() -> None:
    host: Optional[Host] = None
    cap: Optional[cv2.VideoCapture] = None

    try:
        print(f"Loading YOLO model: {MODEL_PATH}")
        print(f"Using device: {DEVICE} (half precision: {USE_HALF})")
        model = YOLO(MODEL_PATH)

        host = Host(HOST_IP, USERNAME, PASSWORD, port=RTSP_PORT)
        stream_url = await get_stream_url(host, CHANNEL, STREAM_TYPE)

        if not stream_url:
            print("Error: Could not get stream URL")
            return

        print(f"Stream URL: {stream_url}")
        cap = open_stream(stream_url)
        if not cap.isOpened():
            print("Error: Could not open RTSP stream in OpenCV")
            return

        print("Starting live detection. Press 'q' to quit.")
        run_detection_loop(cap, model)

    except Exception as err:
        print(f"Error: {err}")
        import traceback

        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        if host is not None:
            try:
                await host.logout()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
