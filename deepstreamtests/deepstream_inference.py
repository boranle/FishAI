#!/usr/bin/env python3
"""
NVIDIA DeepStream Inference Pipeline for Jetson Orin Nano
JetPack 6.2 Compatible

Requirements:
- JetPack 6.2 with DeepStream SDK installed
- GStreamer and dependencies
- Python bindings for DeepStream

Install DeepStream on Jetson:
sudo apt-get update
sudo apt-get install nvidia-jetpack
sudo apt-get install deepstream-6.3
"""

import sys
import os
from typing import Optional, Callable
import gi
import ctypes

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject

# DeepStream Python bindings
sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
sys.path.append("/opt/nvidia/deepstream/deepstream/sources/python")
from pyds import *

# Configuration
RTSP_URI = "rtsp://192.168.1.100:554/h264Preview_01_main"
DETECT_MODEL = "resnet10"  # or "yolov3", "yolov4", etc.
CONF_THRESHOLD = 0.5
OUTPUT_VIDEO = False
VIDEO_OUTPUT_PATH = "/tmp/deepstream_output.mkv"

# DeepStream config paths (adjust for your Jetson)
PGIE_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/source30_primary_detector_nano.txt"
TRACKER_CONFIG_FILE = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/tracker_config.yml"
LABELS_PATH = "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/labels.txt"


class DeepStreamPipeline:
    """Main DeepStream pipeline class for Jetson inference."""
    
    def __init__(self) -> None:
        self.pipeline: Optional[Gst.Pipeline] = None
        self.bus: Optional[Gst.Bus] = None
        self.loop: Optional[GObject.MainLoop] = None
        self.detections_count = 0
        
    def create_element(
        self, 
        factory_name: str, 
        element_name: str
    ) -> Optional[Gst.Element]:
        """Create a GStreamer element."""
        element = Gst.ElementFactory.make(factory_name, element_name)
        if not element:
            print(f"Failed to create element: {element_name} ({factory_name})")
            return None
        return element
    
    def add_and_link_element(
        self,
        pipeline: Gst.Pipeline,
        prev_element: Optional[Gst.Element],
        factory_name: str,
        element_name: str
    ) -> Optional[Gst.Element]:
        """Create element, add to pipeline, and link to previous element."""
        element = self.create_element(factory_name, element_name)
        if not element:
            return None
        
        pipeline.add(element)
        if prev_element:
            if not prev_element.link(element):
                print(f"Failed to link {prev_element.get_name()} -> {element_name}")
                return None
        
        return element
    
    def bus_call(self, bus: Gst.Bus, message: Gst.Message, loop: GObject.MainLoop) -> bool:
        """Handle GStreamer bus messages."""
        message_type = message.type
        
        if message_type == Gst.MessageType.EOS:
            print("End-Of-Stream reached.")
            loop.quit()
        elif message_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err.message}")
            print(f"Debug: {debug}")
            loop.quit()
        elif message_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"Warning: {err.message}")
        
        return True
    
    def osd_sink_pad_buffer_probe(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo
    ) -> Gst.PadProbeReturn:
        """
        Callback on OSD sink pad.
        This is called for every frame in the pipeline.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        # Access frame metadata
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        
        while l_frame:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            
            # Process frame metadata
            l_obj = frame_meta.obj_meta_list
            obj_count = 0
            
            while l_obj:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                # Print detection info
                if obj_meta.confidence > CONF_THRESHOLD:
                    print(f"  Object {obj_count}: Class={obj_meta.class_id}, "
                          f"Confidence={obj_meta.confidence:.2f}, "
                          f"Bbox=({obj_meta.rect_params.left}, "
                          f"{obj_meta.rect_params.top}, "
                          f"{obj_meta.rect_params.width}, "
                          f"{obj_meta.rect_params.height})")
                    obj_count += 1
                    self.detections_count += 1
                
                l_obj = l_obj.next
            
            print(f"Frame {frame_meta.frame_num}: {obj_count} detections")
            l_frame = l_frame.next
        
        return Gst.PadProbeReturn.OK
    
    def build_pipeline(self) -> bool:
        """Build the DeepStream inference pipeline."""
        print("Building DeepStream pipeline...")
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Create pipeline
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            print("Failed to create GStreamer pipeline")
            return False
        
        # Create elements
        print("Creating GStreamer elements...")
        
        # Source
        source = self.add_and_link_element(
            self.pipeline, None, "rtspsrc", "rtsp_src"
        )
        if not source:
            return False
        source.set_property("location", RTSP_URI)
        source.set_property("protocols", "tcp")
        source.set_property("latency", 100)
        
        # RTP depayloaders
        rtp_h264 = self.add_and_link_element(
            self.pipeline, source, "rtph264depay", "rtp_h264"
        )
        if not rtp_h264:
            return False
        
        # H264 parser
        h264_parser = self.add_and_link_element(
            self.pipeline, rtp_h264, "h264parse", "h264_parser"
        )
        if not h264_parser:
            return False
        
        # NVIDIA Video Decoder
        nvvideo_decode = self.add_and_link_element(
            self.pipeline, h264_parser, "nvv4l2decoder", "nvvideo_decode"
        )
        if not nvvideo_decode:
            return False
        nvvideo_decode.set_property("bufapi-version", 1)
        
        # Stream muxer
        streammux = self.add_and_link_element(
            self.pipeline, nvvideo_decode, "nvstreammux", "Stream_muxer"
        )
        if not streammux:
            return False
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 4000000)
        
        # Primary GIE (GPU Inference Engine)
        pgie = self.add_and_link_element(
            self.pipeline, streammux, "nvinfer", "primary-inference"
        )
        if not pgie:
            return False
        pgie.set_property("config-file-path", PGIE_CONFIG_FILE)
        pgie.set_property("batch-size", 1)
        
        # Object tracker
        tracker = self.add_and_link_element(
            self.pipeline, pgie, "nvtracker", "tracker"
        )
        if not tracker:
            return False
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 480)
        tracker.set_property("ll-config-file", TRACKER_CONFIG_FILE)
        tracker.set_property("ll-lib-file", 
                            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
        
        # On-Screen Display
        nvosd = self.add_and_link_element(
            self.pipeline, tracker, "nvdsosd", "onscreendisplay"
        )
        if not nvosd:
            return False
        nvosd.set_property("process-mode", 0)  # GPU mode
        nvosd.set_property("display-text", 1)
        
        # Video converter for encoding
        nvvidconv = self.add_and_link_element(
            self.pipeline, nvosd, "nvvideoconvert", "convertor"
        )
        if not nvvidconv:
            return False
        
        # Capsfilter
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGBA"
        )
        capsfilter = self.add_and_link_element(
            self.pipeline, nvvidconv, "capsfilter", "filter"
        )
        if not capsfilter:
            return False
        capsfilter.set_property("caps", caps)
        
        # Output sink (display or file)
        if OUTPUT_VIDEO:
            # H264 encoder
            encoder = self.add_and_link_element(
                self.pipeline, capsfilter, "nvv4l2h264enc", "encoder"
            )
            if not encoder:
                return False
            encoder.set_property("bitrate", 4000)
            encoder.set_property("preset-level", 1)
            
            # H264 parser
            parser = self.add_and_link_element(
                self.pipeline, encoder, "h264parse", "parser"
            )
            if not parser:
                return False
            
            # Container
            mux = self.add_and_link_element(
                self.pipeline, parser, "qtmux", "mux"
            )
            if not mux:
                return False
            
            # File sink
            filesink = self.add_and_link_element(
                self.pipeline, mux, "filesink", "file_sink"
            )
            if not filesink:
                return False
            filesink.set_property("location", VIDEO_OUTPUT_PATH)
        else:
            # EGL sink for display
            sink = self.add_and_link_element(
                self.pipeline, capsfilter, "nveglglessink", "nvvideo_renderer"
            )
            if not sink:
                return False
            sink.set_property("sync", 0)
        
        # Create bus
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.loop = GObject.MainLoop()
        self.bus.connect("message", self.bus_call, self.loop)
        
        # Add probe on OSD sink pad for object detection callback
        osd_sink_pad = nvosd.get_static_pad("sink")
        if osd_sink_pad:
            osd_sink_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self.osd_sink_pad_buffer_probe
            )
        
        print("Pipeline built successfully!")
        return True
    
    def run(self) -> bool:
        """Start the pipeline."""
        if not self.pipeline:
            print("Pipeline not created")
            return False
        
        print("Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to set pipeline to PLAYING state")
            return False
        
        print("Pipeline running. Press Ctrl+C to stop...")
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("Stopping pipeline...")
        
        return True
    
    def cleanup(self) -> None:
        """Stop and cleanup pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        print(f"Pipeline stopped. Total detections: {self.detections_count}")


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("NVIDIA DeepStream Inference - Jetson Orin Nano")
    print("JetPack 6.2")
    print("=" * 60)
    
    # Check if DeepStream is installed
    deepstream_lib_path = "/opt/nvidia/deepstream/deepstream/lib"
    if not os.path.exists(deepstream_lib_path):
        print(f"Error: DeepStream library not found at {deepstream_lib_path}")
        print("Please install DeepStream SDK:")
        print("  sudo apt-get install nvidia-jetpack")
        print("  sudo apt-get install deepstream-6.3")
        return 1
    
    # Create and run pipeline
    pipeline = DeepStreamPipeline()
    
    if not pipeline.build_pipeline():
        print("Failed to build pipeline")
        return 1
    
    if not pipeline.run():
        print("Failed to run pipeline")
        return 1
    
    pipeline.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
