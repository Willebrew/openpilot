#!/usr/bin/env python3
"""
Camera Stream Client for openpilot
Runs on your computer to view live camera feed from the device

Usage: python3 camera_stream_client.py <device_ip> [--port PORT]

Requirements:
    pip install opencv-python numpy pillow

Controls:
    q - Quit
    s - Save current frame as screenshot
    f - Toggle FPS display
"""

import argparse
import socket
import struct
import time
import numpy as np
from io import BytesIO
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV not available. Install with: pip install opencv-python")
    print("Falling back to PIL display mode (slower)")


def recv_exact(sock, n):
    """Receive exactly n bytes from socket"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def recv_frame(sock):
    """Receive a frame with length header"""
    # Receive frame length (4 bytes)
    raw_len = recv_exact(sock, 4)
    if not raw_len:
        return None

    frame_len = struct.unpack('>I', raw_len)[0]

    # Receive frame data
    frame_data = recv_exact(sock, frame_len)
    return frame_data


def display_with_opencv(sock, device_ip, port):
    """Display stream using OpenCV"""
    window_name = f"openpilot Camera Stream - {device_ip}:{port}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    show_fps = True
    fps_text = "Connecting..."

    print("Receiving stream... Press 'q' to quit, 's' to save screenshot, 'f' to toggle FPS")

    try:
        while True:
            # Receive frame
            jpeg_data = recv_frame(sock)
            if jpeg_data is None:
                print("Connection closed by server")
                break

            # Decode JPEG
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode frame")
                continue

            # Calculate FPS
            current_time = time.time()
            frame_count += 1

            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                fps_text = f"FPS: {fps:.1f} | Size: {len(jpeg_data)/1024:.1f} KB"
                last_fps_time = current_time

            # Overlay FPS
            if show_fps:
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display frame
            cv2.imshow(window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"FPS display: {'ON' if show_fps else 'OFF'}")

    finally:
        cv2.destroyAllWindows()


def display_with_pil(sock, device_ip, port):
    """Display stream using PIL (fallback, slower)"""
    print(f"Receiving stream from {device_ip}:{port}...")
    print("Note: PIL mode is slower. Install OpenCV for better performance:")
    print("  pip install opencv-python")
    print("\nPress Ctrl+C to quit")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Receive frame
            jpeg_data = recv_frame(sock)
            if jpeg_data is None:
                print("Connection closed by server")
                break

            # Decode and display
            img = Image.open(BytesIO(jpeg_data))
            img.show()

            frame_count += 1
            current_time = time.time()
            fps = frame_count / (current_time - start_time)

            if frame_count % 30 == 0:
                print(f"Received {frame_count} frames | FPS: {fps:.1f} | Size: {len(jpeg_data)/1024:.1f} KB")

    except KeyboardInterrupt:
        print("\nQuitting...")


def main():
    parser = argparse.ArgumentParser(description="View openpilot camera stream")
    parser.add_argument('device_ip', help='IP address of the openpilot device')
    parser.add_argument('--port', type=int, default=5555, help='TCP port (default: 5555)')
    args = parser.parse_args()

    print("=" * 60)
    print("openpilot Camera Stream Client")
    print("=" * 60)
    print(f"Connecting to: {args.device_ip}:{args.port}")
    print("=" * 60)

    # Connect to server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.device_ip, args.port))
        print(f"Connected to server at {args.device_ip}:{args.port}")
    except ConnectionRefusedError:
        print(f"Error: Could not connect to {args.device_ip}:{args.port}")
        print("Make sure the server is running on the device:")
        print(f"  python3 camera_stream_server.py --port {args.port}")
        return
    except socket.gaierror:
        print(f"Error: Invalid IP address: {args.device_ip}")
        return

    try:
        if HAS_CV2:
            display_with_opencv(sock, args.device_ip, args.port)
        else:
            display_with_pil(sock, args.device_ip, args.port)
    finally:
        sock.close()
        print("Disconnected")


if __name__ == "__main__":
    main()
