#!/usr/bin/env python3
"""
Simple MJPEG Camera Streamer for openpilot
Works reliably with any video player

Usage: python3 simple_stream_server.py [--port PORT] [--camera CAMERA] [--fps FPS]
"""

import argparse
import socket
import subprocess
import time
import cv2
import numpy as np

import cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.system.hardware import PC
from openpilot.system.manager.process_config import managed_processes


VISION_STREAMS = {
    "road": VisionStreamType.VISION_STREAM_ROAD,
    "driver": VisionStreamType.VISION_STREAM_DRIVER,
    "wide": VisionStreamType.VISION_STREAM_WIDE_ROAD,
}

CAMERA_SERVICES = {
    "road": "roadCameraState",
    "driver": "driverCameraState",
    "wide": "wideRoadCameraState",
}


def yuv_to_bgr(buf):
    """Convert YUV NV12 to BGR using OpenCV (fast, hardware accelerated)"""
    h, w = buf.height, buf.width

    # NV12 format: full Y plane, then interleaved UV at half resolution
    # Extract Y plane - accounting for stride and padding
    y_plane = np.frombuffer(buf.data, dtype=np.uint8, count=buf.uv_offset)
    # Calculate actual rows (may include padding rows)
    y_rows = len(y_plane) // buf.stride
    y = y_plane.reshape((y_rows, buf.stride))[:h, :w].copy()

    # Extract UV plane - accounting for stride
    # UV is interleaved (UVUVUV...) at half resolution
    uv_size = (h // 2) * buf.stride
    uv_plane = np.frombuffer(buf.data, dtype=np.uint8, offset=buf.uv_offset, count=uv_size)
    uv = uv_plane.reshape((h // 2, buf.stride))[:, :w].copy()

    # Create properly formatted NV12 image for OpenCV
    # Stack Y and UV vertically: [Y: h×w] [UV: h/2×w]
    yuv_nv12 = np.vstack([y, uv])

    # Convert NV12 to BGR using OpenCV (hardware accelerated)
    bgr = cv2.cvtColor(yuv_nv12, cv2.COLOR_YUV2BGR_NV12)

    return bgr


def stream_mjpeg(camera_name, port, quality, target_fps):
    """Stream camera as MJPEG over HTTP"""
    print(f"Starting MJPEG stream server for '{camera_name}' camera")

    # Check if camerad is already running
    camerad_running = False
    try:
        subprocess.check_call(["pgrep", "camerad"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        camerad_running = True
    except subprocess.CalledProcessError:
        print("Starting camerad...")
        if not PC:
            managed_processes['camerad'].start()
            time.sleep(2)

    # Setup VisionIPC
    service_name = CAMERA_SERVICES[camera_name]
    stream_type = VISION_STREAMS[camera_name]

    sm = messaging.SubMaster([service_name])
    vipc_client = VisionIpcClient("camerad", stream_type, True)

    # Wait for camera
    print("Waiting for camera...")
    while sm[service_name].frameId < 10:
        sm.update(100)

    vipc_client.connect(True)
    print("Camera ready!")

    # Setup HTTP server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', port))
    server_sock.listen(1)

    print(f"MJPEG server listening on http://0.0.0.0:{port}")
    print(f"\nTo view stream, open in your browser:")
    print(f"  http://<device_ip>:{port}")
    print(f"\nOr use VLC/ffplay:")
    print(f"  ffplay http://<device_ip>:{port}")
    print(f"  vlc http://<device_ip>:{port}")

    frame_time = 1.0 / target_fps
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    try:
        while True:
            client_sock, addr = server_sock.accept()
            print(f"\nClient connected from {addr}")

            # Send HTTP headers for MJPEG stream
            headers = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Connection: close\r\n"
                b"\r\n"
            )
            try:
                client_sock.sendall(headers)
            except:
                client_sock.close()
                continue

            frame_count = 0
            start_time = time.time()
            last_fps_time = start_time
            last_frame_time = start_time

            try:
                while True:
                    current_time = time.time()

                    # Rate limit to target FPS
                    if current_time - last_frame_time < frame_time:
                        time.sleep(0.001)
                        continue

                    last_frame_time = current_time

                    # Get frame from camera
                    buf = vipc_client.recv()
                    if buf is None:
                        time.sleep(0.01)
                        continue

                    # Convert YUV to BGR (fast, hardware accelerated)
                    bgr = yuv_to_bgr(buf)

                    # Encode to JPEG (fast, OpenCV uses libjpeg-turbo)
                    _, jpeg = cv2.imencode('.jpg', bgr, encode_params)
                    jpeg_bytes = jpeg.tobytes()

                    # Send as MJPEG frame
                    frame_header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n"
                        b"\r\n"
                    )

                    try:
                        client_sock.sendall(frame_header + jpeg_bytes + b"\r\n")
                        frame_count += 1

                        # Print stats
                        if current_time - last_fps_time >= 1.0:
                            fps = frame_count / (current_time - start_time)
                            print(f"Streaming at {fps:.1f} FPS | Frame size: {len(jpeg_bytes)/1024:.1f} KB")
                            last_fps_time = current_time

                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print("Client disconnected")
                        break

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            finally:
                client_sock.close()

    finally:
        server_sock.close()
        if not camerad_running and not PC:
            managed_processes['camerad'].stop()
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Stream openpilot camera as MJPEG")
    parser.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')
    parser.add_argument('--camera', choices=['road', 'driver', 'wide'], default='road',
                        help='Camera to stream (default: road)')
    parser.add_argument('--quality', type=int, default=60, help='JPEG quality 1-100 (default: 60)')
    parser.add_argument('--fps', type=int, default=20, help='Target FPS (default: 20)')
    args = parser.parse_args()

    print("=" * 60)
    print("openpilot MJPEG Stream Server")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Port: {args.port}")
    print(f"Quality: {args.quality}")
    print(f"Target FPS: {args.fps}")
    print("=" * 60)

    stream_mjpeg(args.camera, args.port, args.quality, args.fps)


if __name__ == "__main__":
    main()
