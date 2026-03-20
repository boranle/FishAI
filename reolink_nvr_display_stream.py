# ffmpeg/ffplay is required to run this script. Install FFmpeg from https://ffmpeg.org/download.html and ensure ffplay is in your system PATH.

from reolink_aio.api import Host
import asyncio
import shutil
import subprocess

# Stream display configuration
HOST_IP = '192.168.1.100'
USERNAME = 'admin'
PASSWORD = 'Fishfish'
RTSP_PORT = 554
CHANNEL = 0  # Camera channel
STREAM_TYPE = 'main'  # 'main' or 'sub'

def display_with_ffplay(stream_url: str) -> bool:
    """Display the RTSP stream using ffplay."""
    ffplay_path = shutil.which("ffplay")
    if ffplay_path is None:
        print("Error: ffplay was not found in PATH.")
        print("Install FFmpeg (includes ffplay) and run this script again.")
        return False

    print("\nLaunching ffplay... close the ffplay window to stop.")
    cmd = [
        ffplay_path,
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-rtsp_transport",
        "tcp",
        stream_url,
    ]

    try:
        subprocess.run(cmd, check=False)
        return True
    except Exception as err:
        print(f"Failed to launch ffplay fallback: {err}")
        return False


async def main():
    host = None
    try:
        # Initialize the host
        host = Host(HOST_IP, USERNAME, PASSWORD, port=RTSP_PORT)
        print("Connecting to NVR...")
        
        # Connect and obtain/cache device settings and capabilities
        await host.get_host_data()
        print(f"Connected successfully!")
        print(f"Is NVR: {host.is_nvr}, Number of channels: {host.num_channels}")
        print(f"Available stream channels: {host.stream_channels}")

        if CHANNEL not in host.stream_channels:
            print(f"Error: channel {CHANNEL} is not available on this device")
            return

        print(f"Camera name: {host.camera_name(CHANNEL)}")
        
        # Check if RTSP is available
        if "RTSP" not in host.capabilities.get("Host", set()):
            print("Warning: RTSP may not be available on this device")
        
        # Get the RTSP stream URL
        print(f"\nGetting RTSP stream source for channel {CHANNEL}...")
        stream_url = await host.get_rtsp_stream_source(CHANNEL, stream=STREAM_TYPE)
        
        if stream_url is None:
            print("Error: Could not get stream URL")
            return
        
        print(f"Stream URL: {stream_url}")

        if not display_with_ffplay(stream_url):
            print("Could not display the stream with ffplay.")
            return

        print("Disconnected from NVR")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if host is not None:
            try:
                await host.logout()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
