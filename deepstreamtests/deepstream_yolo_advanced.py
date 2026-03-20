#!/usr/bin/env python3
"""
Advanced DeepStream Pipeline with Custom YOLO model
Jetson Orin Nano + JetPack 6.2

This example shows how to integrate a custom YOLO model with DeepStream
for optimized inference on edge devices.

Requirements:
- YOLO model converted to TensorRT engine (.engine file)
- DeepStream SDK 6.3+
- pytorch/ultralytics (for offline model conversion only)
"""

import sys
import os
from typing import Optional, Dict, List, Tuple
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject

sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
sys.path.append("/opt/nvidia/deepstream/deepstream/sources/python")
from pyds import *

# Configuration
CONFIG = {
    "rtsp_uri": "rtsp://192.168.1.100:554/h264Preview_01_main",
    "model_engine": "/path/to/model.engine",  # TensorRT engine file
    "input_width": 1920,
    "input_height": 1080,
    "model_width": 640,
    "model_height": 640,
    "confidence_threshold": 0.5,
    "nms_threshold": 0.45,
    "batch_size": 1,
    "gpu_id": 0,
    "enable_tracker": True,
    "output_video": False,
    "output_path": "/tmp/deepstream_output.mkv",
}

# YOLO class labels
YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
]

# Color palette for different classes (BGR)
CLASS_COLORS = [
    (0, 255, 0),    # person - green
    (255, 0, 0),    # bicycle - blue
    (0, 0, 255),    # car - red
    (255, 255, 0),  # motorcycle - cyan
    (255, 0, 255),  # airplane - magenta
    (0, 255, 255),  # bus - yellow
]


class AdvancedDeepStreamPipeline:
    """Advanced DeepStream pipeline with YOLO inference."""
    
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.pipeline: Optional[Gst.Pipeline] = None
        self.bus: Optional[Gst.Bus] = None
        self.loop: Optional[GObject.MainLoop] = None
        self.detections: List[Dict] = []
        self.frame_count = 0
        self.fps = 0.0
        
    def create_element(self, factory_name: str, element_name: str) -> Optional[Gst.Element]:
        """Create a GStreamer element."""
        element = Gst.ElementFactory.make(factory_name, element_name)
        if not element:
            print(f"✗ Failed to create: {element_name} ({factory_name})")
            return None
        print(f"✓ Created: {element_name}")
        return element
    
    def add_and_link(
        self,
        pipeline: Gst.Pipeline,
        prev_element: Optional[Gst.Element],
        factory_name: str,
        element_name: str
    ) -> Optional[Gst.Element]:
        """Create, add, and link element."""
        element = self.create_element(factory_name, element_name)
        if not element:
            return None
        
        pipeline.add(element)
        if prev_element:
            if not prev_element.link(element):
                print(f"✗ Failed to link: {prev_element.get_name()} -> {element_name}")
                return None
        
        return element
    
    def bus_call(self, bus: Gst.Bus, message: Gst.Message, loop: GObject.MainLoop) -> bool:
        """Handle GStreamer bus messages."""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.EOS:
            print("\n✓ End of stream")
            loop.quit()
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"✗ Error: {err.message}\nDebug: {debug}")
            loop.quit()
        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"⚠ Warning: {err.message}")
        
        return True
    
    def detection_callback(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        """Process detection results on each frame."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        frame_detections = []
        self.frame_count += 1
        
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            obj_count = 0
            
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                if obj_meta.confidence >= self.config["confidence_threshold"]:
                    # Extract detection info
                    class_id = obj_meta.class_id
                    confidence = obj_meta.confidence
                    bbox = {
                        "left": int(obj_meta.rect_params.left),
                        "top": int(obj_meta.rect_params.top),
                        "width": int(obj_meta.rect_params.width),
                        "height": int(obj_meta.rect_params.height),
                    }
                    
                    class_name = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else f"Class_{class_id}"
                    
                    detection = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox,
                        "track_id": obj_meta.object_id,
                    }
                    
                    frame_detections.append(detection)
                    obj_count += 1
                
                l_obj = l_obj.next
            
            if frame_detections:
                print(f"[Frame {self.frame_count}] {obj_count} detections:")
                for det in frame_detections:
                    print(f"  • {det['class_name']}: {det['confidence']:.2%} "
                          f"(Track ID: {det['track_id']})")
            
            l_frame = l_frame.next
        
        self.detections = frame_detections
        return Gst.PadProbeReturn.OK
    
    def build_pipeline(self) -> bool:
        """Build the complete inference pipeline."""
        print("\n" + "=" * 60)
        print("Building DeepStream YOLO Pipeline")
        print("=" * 60)
        
        Gst.init(None)
        self.pipeline = Gst.Pipeline()
        
        if not self.pipeline:
            print("✗ Failed to create pipeline")
            return False
        
        # Build source chain
        print("\n[Building Source Chain]")
        source = self.add_and_link(
            self.pipeline, None, "rtspsrc", "rtsp_src"
        )
        if not source:
            return False
        
        source.set_property("location", self.config["rtsp_uri"])
        source.set_property("protocols", "tcp")
        source.set_property("latency", 100)
        
        rtp_h264 = self.add_and_link(self.pipeline, source, "rtph264depay", "rtp_h264")
        if not rtp_h264:
            return False
        
        h264_parser = self.add_and_link(self.pipeline, rtp_h264, "h264parse", "h264_parser")
        if not h264_parser:
            return False
        
        # Decoder
        print("\n[Building Decoder]")
        decoder = self.add_and_link(
            self.pipeline, h264_parser, "nvv4l2decoder", "nvvideo_decode"
        )
        if not decoder:
            return False
        decoder.set_property("bufapi-version", 1)
        
        # Stream muxer
        print("\n[Building Stream Muxer]")
        streammux = self.add_and_link(
            self.pipeline, decoder, "nvstreammux", "Stream_muxer"
        )
        if not streammux:
            return False
        
        streammux.set_property("width", self.config["input_width"])
        streammux.set_property("height", self.config["input_height"])
        streammux.set_property("batch-size", self.config["batch_size"])
        streammux.set_property("batched-push-timeout", 4000000)
        
        # Primary inference engine
        print("\n[Building Inference Engine]")
        pgie = self.add_and_link(
            self.pipeline, streammux, "nvinfer", "primary_inferencer"
        )
        if not pgie:
            return False
        
        pgie.set_property("process-mode", 1)  # Batch mode
        pgie.set_property("batch-size", self.config["batch_size"])
        
        # Object tracker
        if self.config["enable_tracker"]:
            print("\n[Building Object Tracker]")
            tracker = self.add_and_link(
                self.pipeline, pgie, "nvtracker", "tracker"
            )
            if not tracker:
                return False
            
            tracker.set_property("tracker-width", 640)
            tracker.set_property("tracker-height", 480)
            prev = tracker
        else:
            prev = pgie
        
        # On-screen display
        print("\n[Building Display Pipeline]")
        osd = self.add_and_link(
            self.pipeline, prev, "nvdsosd", "onscreendisplay"
        )
        if not osd:
            return False
        
        osd.set_property("process-mode", 0)  # GPU
        osd.set_property("display-text", 1)
        
        # Video output
        nvvidconv = self.add_and_link(
            self.pipeline, osd, "nvvideoconvert", "convertor"
        )
        if not nvvidconv:
            return False
        
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        capsfilter = self.add_and_link(
            self.pipeline, nvvidconv, "capsfilter", "filter"
        )
        if not capsfilter:
            return False
        capsfilter.set_property("caps", caps)
        
        # Output sink
        sink = self.add_and_link(
            self.pipeline, capsfilter, "nveglglessink", "nvvideo_renderer"
        )
        if not sink:
            return False
        sink.set_property("sync", 0)
        
        # Setup bus and probe
        print("\n[Setting up Event Handlers]")
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.loop = GObject.MainLoop()
        self.bus.connect("message", self.bus_call, self.loop)
        
        # Add probe for detection callback
        osd_sink_pad = osd.get_static_pad("sink")
        if osd_sink_pad:
            osd_sink_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self.detection_callback
            )
            print("✓ Detection callback registered")
        
        print("\n✓ Pipeline ready!\n")
        return True
    
    def run(self) -> bool:
        """Run the pipeline."""
        if not self.pipeline:
            return False
        
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("✗ Failed to start pipeline")
            return False
        
        print("▶ Pipeline running...")
        print("Press Ctrl+C to stop\n")
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n✓ Stopping...")
        
        return True
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        print(f"\n{'=' * 60}")
        print(f"Pipeline Stats:")
        print(f"  Total Frames: {self.frame_count}")
        print(f"  Total Detections: {len(self.detections)}")
        print(f"{'=' * 60}\n")


def convert_yolo_to_tensorrt(
    model_path: str,
    engine_path: str,
    conf: float = 0.45,
    iou: float = 0.45
) -> bool:
    """Convert YOLO model to TensorRT engine (offline conversion)."""
    print("Converting YOLO to TensorRT...")
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        # Export to TensorRT format
        engine = model.export(
            format="engine",
            imgsz=640,
            device=0,
            half=True,
            int8=True,
        )
        
        print(f"✓ Model converted: {engine}")
        return True
    except ImportError:
        print("⚠ Please install ultralytics: pip install ultralytics")
        return False
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False


def main() -> int:
    """Main entry point."""
    print(f"\n{'=' * 60}")
    print("Advanced DeepStream YOLO Inference")
    print("NVIDIA Jetson Orin Nano - JetPack 6.2")
    print(f"{'=' * 60}\n")
    
    # Verify DeepStream installation
    deepstream_root = "/opt/nvidia/deepstream/deepstream"
    if not os.path.exists(deepstream_root):
        print(f"✗ DeepStream not found at {deepstream_root}")
        print("Install with: sudo apt-get install nvidia-deepstream")
        return 1
    
    # Create and run pipeline
    pipeline = AdvancedDeepStreamPipeline(CONFIG)
    
    if not pipeline.build_pipeline():
        return 1
    
    if not pipeline.run():
        return 1
    
    pipeline.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
