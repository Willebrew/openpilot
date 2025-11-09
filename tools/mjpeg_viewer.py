#!/usr/bin/env python3
"""
MJPEG Stream Viewer for Mac
Opens the openpilot camera stream in a standalone window

Usage: python3 mjpeg_viewer.py [device_ip] [--port PORT]

Requirements:
    pip install opencv-python requests numpy
"""

import argparse
import sys
import time
import requests
import cv2
import numpy as np
from io import BytesIO


def stream_mjpeg_viewer(url):
    """View MJPEG stream in OpenCV window"""
    print(f"Connecting to {url}...")

    # Create window
    window_name = f"openpilot Camera Stream - {url}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)

    print("Opening stream...")
    print("\nControls:")
    print("  q - Quit")
    print("  f - Toggle fullscreen")
    print("  s - Save screenshot")
    print("  + - Increase window size")
    print("  - - Decrease window size")
    print("")

    try:
        # Connect to MJPEG stream
        response = requests.get(url, stream=True, timeout=5)

        if response.status_code != 200:
            print(f"Error: Server returned status {response.status_code}")
            return

        print("Connected! Streaming video...")

        frame_count = 0
        start_time = time.time()
        last_fps_time = start_time
        fullscreen = False

        # Read stream byte by byte looking for JPEG boundaries
        bytes_buffer = b''

        for chunk in response.iter_content(chunk_size=1024):
            bytes_buffer += chunk

            # Look for JPEG start and end markers
            a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
            b = bytes_buffer.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]

                # Decode JPEG
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # Calculate FPS
                    frame_count += 1
                    current_time = time.time()

                    if current_time - last_fps_time >= 1.0:
                        fps = frame_count / (current_time - start_time)
                        # Update window title with FPS
                        cv2.setWindowTitle(window_name, f"openpilot Camera - {fps:.1f} FPS")
                        last_fps_time = current_time

                    # Display frame
                    cv2.imshow(window_name, frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        print("Quit requested")
                        break
                    elif key == ord('f'):
                        fullscreen = not fullscreen
                        if fullscreen:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        print(f"Fullscreen: {'ON' if fullscreen else 'OFF'}")
                    elif key == ord('s'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"openpilot_screenshot_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Screenshot saved: {filename}")
                    elif key == ord('+') or key == ord('='):
                        # Increase window size
                        w, h = cv2.getWindowImageRect(window_name)[2:]
                        cv2.resizeWindow(window_name, int(w * 1.2), int(h * 1.2))
                    elif key == ord('-') or key == ord('_'):
                        # Decrease window size
                        w, h = cv2.getWindowImageRect(window_name)[2:]
                        cv2.resizeWindow(window_name, int(w * 0.8), int(h * 0.8))

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {url}")
        print("Make sure the server is running on the device:")
        print("  python3 tools/simple_stream_server.py")
    except requests.exceptions.Timeout:
        print(f"Error: Connection timeout to {url}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cv2.destroyAllWindows()
        print("Viewer closed")


def main():
    parser = argparse.ArgumentParser(description="View openpilot MJPEG camera stream")
    parser.add_argument('device_ip', nargs='?', default='localhost',
                        help='IP address of the openpilot device (default: localhost)')
    parser.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')
    args = parser.parse_args()

    url = f"http://{args.device_ip}:{args.port}"

    print("=" * 60)
    print("openpilot MJPEG Stream Viewer")
    print("=" * 60)
    print(f"Stream URL: {url}")
    print("=" * 60)

    stream_mjpeg_viewer(url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
