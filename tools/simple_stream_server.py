#!/usr/bin/env python3
"""
Advanced MJPEG Camera Streamer for openpilot with Model Overlays
Supports live camera switching and streams model predictions

Usage: python3 simple_stream_server.py [--port PORT] [--fps FPS] [--quality QUALITY]
"""

import argparse
import socket
import subprocess
import time
import cv2
import numpy as np
import json
from urllib.parse import parse_qs, urlparse

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


def extract_model_data(sm):
    """Extract relevant model data for overlay rendering"""
    model_data = {}

    # Get modelV2 message
    if sm.updated['modelV2']:
        m = sm['modelV2']

        # Extract lane lines (4 lines with x, y, z arrays)
        model_data['laneLines'] = []
        for i, lane in enumerate(m.laneLines):
            if i < len(m.laneLineProbs):
                model_data['laneLines'].append({
                    'x': lane.x[:],
                    'y': lane.y[:],
                    'z': lane.z[:],
                    'prob': m.laneLineProbs[i]
                })

        # Extract road edges (2 edges)
        model_data['roadEdges'] = []
        for edge in m.roadEdges:
            model_data['roadEdges'].append({
                'x': edge.x[:],
                'y': edge.y[:],
                'z': edge.z[:]
            })

        # Extract lead cars (up to 3)
        model_data['leads'] = []
        for lead in m.leadsV3:
            if lead.prob > 0.3:  # Only include confident detections
                model_data['leads'].append({
                    'x': lead.x[:],
                    'y': lead.y[:],
                    'prob': lead.prob
                })

        # Extract vehicle path
        model_data['path'] = {
            'x': m.position.x[:],
            'y': m.position.y[:],
            'z': m.position.z[:]
        }

    # Get calibration data
    if sm.updated['liveCalibration']:
        cal = sm['liveCalibration']
        model_data['calibration'] = {
            'rpy': cal.rpyCalib[:],
            'valid': cal.calStatus == 1
        }

    return model_data


def stream_mjpeg(port, quality, target_fps):
    """Stream camera as MJPEG over HTTP with model overlay data"""
    print("Starting advanced MJPEG stream server")

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

    # Check if modeld is already running
    modeld_running = False
    try:
        subprocess.check_call(["pgrep", "modeld"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        modeld_running = True
    except subprocess.CalledProcessError:
        print("Starting modeld...")
        if not PC:
            managed_processes['modeld'].start()
            time.sleep(2)

    # Setup VisionIPC clients for all cameras
    vipc_clients = {}
    for cam_name, stream_type in VISION_STREAMS.items():
        vipc_clients[cam_name] = VisionIpcClient("camerad", stream_type, True)

    # Setup messaging - subscribe to all cameras and model
    services = list(CAMERA_SERVICES.values()) + ['modelV2', 'liveCalibration']
    sm = messaging.SubMaster(services)

    # Wait for cameras and model
    print("Waiting for cameras and model...")
    for service_name in CAMERA_SERVICES.values():
        while sm[service_name].frameId < 10:
            sm.update(100)

    # Connect all cameras
    for client in vipc_clients.values():
        client.connect(True)

    print("All cameras ready!")
    print("Waiting for model...")
    while not sm.updated['modelV2']:
        sm.update(100)
    print("Model ready!")

    # Setup HTTP server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', port))
    server_sock.listen(1)

    print(f"\nMJPEG server listening on http://0.0.0.0:{port}")
    print(f"\nTo view stream:")
    print(f"  http://<device_ip>:{port}?camera=road")
    print(f"  http://<device_ip>:{port}?camera=driver")
    print(f"  http://<device_ip>:{port}?camera=wide")

    frame_time = 1.0 / target_fps
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    try:
        while True:
            client_sock, addr = server_sock.accept()

            # Read HTTP request to get camera parameter
            try:
                request = client_sock.recv(1024).decode('utf-8', errors='ignore')
                request_line = request.split('\r\n')[0]
                path = request_line.split(' ')[1]
                query = parse_qs(urlparse(path).query)
                camera_name = query.get('camera', ['road'])[0]

                if camera_name not in VISION_STREAMS:
                    camera_name = 'road'

                print(f"\nClient connected from {addr} - Camera: {camera_name}")
            except:
                camera_name = 'road'
                print(f"\nClient connected from {addr} - Default camera: {camera_name}")

            # Get the VisionIPC client for selected camera
            vipc_client = vipc_clients[camera_name]

            # Send HTTP headers for multipart stream (JPEG + JSON)
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

                    # Update messaging
                    sm.update(0)

                    # Get frame from selected camera
                    buf = vipc_client.recv()
                    if buf is None:
                        time.sleep(0.01)
                        continue

                    # Convert YUV to BGR
                    bgr = yuv_to_bgr(buf)

                    # Encode to JPEG
                    _, jpeg = cv2.imencode('.jpg', bgr, encode_params)
                    jpeg_bytes = jpeg.tobytes()

                    # Extract model data
                    model_data = extract_model_data(sm)
                    model_json = json.dumps(model_data).encode('utf-8')

                    # Send multipart frame with both image and model data
                    frame_header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n"
                        b"X-Camera: " + camera_name.encode() + b"\r\n"
                        b"X-Model-Data: " + model_json + b"\r\n"
                        b"\r\n"
                    )

                    try:
                        client_sock.sendall(frame_header + jpeg_bytes + b"\r\n")
                        frame_count += 1

                        # Print stats
                        if current_time - last_fps_time >= 5.0:
                            fps = frame_count / (current_time - start_time)
                            print(f"[{camera_name}] Streaming at {fps:.1f} FPS | JPEG: {len(jpeg_bytes)/1024:.1f} KB | Model: {len(model_json)/1024:.1f} KB")
                            last_fps_time = current_time

                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print(f"Client disconnected")
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
        if not modeld_running and not PC:
            managed_processes['modeld'].stop()
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Stream openpilot camera + model as MJPEG")
    parser.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')
    parser.add_argument('--quality', type=int, default=60, help='JPEG quality 1-100 (default: 60)')
    parser.add_argument('--fps', type=int, default=20, help='Target FPS (default: 20)')
    args = parser.parse_args()

    print("=" * 60)
    print("openpilot Advanced MJPEG Stream Server")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Quality: {args.quality}")
    print(f"Target FPS: {args.fps}")
    print(f"Features: Live camera switching + Model overlays")
    print("=" * 60)

    stream_mjpeg(args.port, args.quality, args.fps)


if __name__ == "__main__":
    main()
