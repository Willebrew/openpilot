#!/usr/bin/env python3
"""
Camera Stream Server for openpilot
Runs on the device to stream live camera feed over TCP

Usage: python3 camera_stream_server.py [--port PORT] [--camera CAMERA] [--quality QUALITY]
"""

import argparse
import socket
import struct
import subprocess
import time
import numpy as np
from io import BytesIO
from PIL import Image

import cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.params import Params
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


def yuv_to_rgb(y, u, v):
    """Convert YUV420 to RGB"""
    ul = np.repeat(np.repeat(u, 2).reshape(u.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)
    vl = np.repeat(np.repeat(v, 2).reshape(v.shape[0], y.shape[1]), 2, axis=0).reshape(y.shape)

    yuv = np.dstack((y, ul, vl)).astype(np.int16)
    yuv[:, :, 1:] -= 128

    m = np.array([
        [1.00000,  1.00000, 1.00000],
        [0.00000, -0.39465, 2.03211],
        [1.13983, -0.58060, 0.00000],
    ])
    rgb = np.dot(yuv, m).clip(0, 255)
    return rgb.astype(np.uint8)


def extract_image(buf):
    """Extract RGB image from VisionIPC buffer"""
    y = np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
    u = np.array(buf.data[buf.uv_offset::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]
    v = np.array(buf.data[buf.uv_offset+1::2], dtype=np.uint8).reshape((-1, buf.stride//2))[:buf.height//2, :buf.width//2]

    return yuv_to_rgb(y, u, v)


def send_frame(sock, frame_data):
    """Send a frame with length header"""
    # Send frame length (4 bytes)
    sock.sendall(struct.pack('>I', len(frame_data)))
    # Send frame data
    sock.sendall(frame_data)


def stream_camera(camera_name, port, quality):
    """Stream camera feed over TCP socket"""
    print(f"Starting camera stream server for '{camera_name}' camera on port {port}")

    # Setup messaging and VisionIPC
    service_name = CAMERA_SERVICES[camera_name]
    stream_type = VISION_STREAMS[camera_name]

    # Check if camerad is already running
    camerad_running = False
    try:
        subprocess.check_call(["pgrep", "camerad"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("camerad already running")
        camerad_running = True
    except subprocess.CalledProcessError:
        print("camerad not running, starting it...")
        if not PC:
            try:
                managed_processes['camerad'].start()
                print("camerad started successfully")
                time.sleep(2)  # Give camerad time to initialize
            except Exception as e:
                print(f"Warning: Could not start camerad: {e}")
                print("Attempting to continue anyway...")
        else:
            print("Running on PC, camerad may need to be started manually")

    sm = messaging.SubMaster([service_name])
    vipc_client = VisionIpcClient("camerad", stream_type, True)

    # Setup TCP server FIRST so clients can connect while we wait
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', port))
    server_sock.listen(1)
    server_sock.settimeout(1.0)  # Non-blocking accept with 1s timeout

    print(f"Server listening on 0.0.0.0:{port}")
    print(f"Waiting for camera to start and client connection...")

    try:
        # Wait for camera to be ready (in background while accepting connections)
        camera_ready = False
        max_wait = 30  # 30 seconds timeout
        wait_start = time.time()

        while not camera_ready:
            sm.update(100)  # 100ms timeout
            if sm[service_name].frameId >= 10:
                camera_ready = True
                print("Camera ready!")
                break

            if time.time() - wait_start > max_wait:
                print(f"Warning: Camera not ready after {max_wait}s, but continuing...")
                print(f"Current frameId: {sm[service_name].frameId}")
                camera_ready = True  # Continue anyway
                break

        # Connect to VisionIPC
        print("Connecting to camera stream...")
        vipc_client.connect(True)
        print("Connected to VisionIPC")

        print("Ready for client connections!")

        while True:
            # Accept client connection (non-blocking with timeout)
            try:
                client_sock, addr = server_sock.accept()
                print(f"Client connected from {addr}")
            except socket.timeout:
                continue  # No client yet, keep waiting

            try:
                frame_count = 0
                start_time = time.time()
                last_fps_time = start_time

                while True:
                    # Receive frame from camera
                    buf = vipc_client.recv()
                    if buf is None:
                        print("Failed to receive frame")
                        time.sleep(0.01)
                        continue

                    # Extract and convert to RGB
                    rgb_img = extract_image(buf)

                    # Convert to JPEG
                    img = Image.fromarray(rgb_img)
                    jpeg_buffer = BytesIO()
                    img.save(jpeg_buffer, format='JPEG', quality=quality)
                    jpeg_data = jpeg_buffer.getvalue()

                    # Send to client
                    try:
                        send_frame(client_sock, jpeg_data)
                        frame_count += 1

                        # Print FPS every second
                        current_time = time.time()
                        if current_time - last_fps_time >= 1.0:
                            fps = frame_count / (current_time - start_time)
                            print(f"Streaming at {fps:.1f} FPS | Frame size: {len(jpeg_data)/1024:.1f} KB")
                            last_fps_time = current_time
                    except (BrokenPipeError, ConnectionResetError):
                        print("Client disconnected")
                        break

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            finally:
                client_sock.close()

    finally:
        server_sock.close()
        # Stop camerad if we started it
        if not camerad_running and not PC:
            print("Stopping camerad...")
            try:
                managed_processes['camerad'].stop()
            except Exception as e:
                print(f"Warning: Could not stop camerad: {e}")
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Stream openpilot camera over TCP")
    parser.add_argument('--port', type=int, default=5555, help='TCP port to listen on (default: 5555)')
    parser.add_argument('--camera', choices=['road', 'driver', 'wide'], default='road',
                        help='Camera to stream (default: road)')
    parser.add_argument('--quality', type=int, default=85, help='JPEG quality 1-100 (default: 85)')
    args = parser.parse_args()

    print("=" * 60)
    print("openpilot Camera Stream Server")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Port: {args.port}")
    print(f"Quality: {args.quality}")
    print("=" * 60)

    stream_camera(args.camera, args.port, args.quality)


if __name__ == "__main__":
    main()
