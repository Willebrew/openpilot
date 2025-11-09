#!/usr/bin/env python3
"""
Advanced MJPEG Stream Viewer with Model Overlays
Opens the openpilot camera stream with live model predictions overlay

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
import json
import threading
from io import BytesIO


# Camera intrinsics (from openpilot hardware config)
CAMERA_CONFIGS = {
    'road': {
        'focal_mm': 8.0,
        'width': 1928,
        'height': 1208,
        'pixel_size_mm': 0.003,
    },
    'wide': {
        'focal_mm': 1.71,
        'width': 1928,
        'height': 1208,
        'pixel_size_mm': 0.003,
    },
    'driver': {
        'focal_mm': 1.71,
        'width': 1928,
        'height': 1208,
        'pixel_size_mm': 0.003,
    }
}


def get_camera_intrinsics(camera_name):
    """Calculate camera intrinsic matrix"""
    cfg = CAMERA_CONFIGS[camera_name]
    focal_px = cfg['focal_mm'] / cfg['pixel_size_mm']

    # Intrinsic matrix K
    K = np.array([
        [focal_px,     0.0, cfg['width'] / 2.0],
        [0.0,     focal_px, cfg['height'] / 2.0],
        [0.0,          0.0, 1.0]
    ])
    return K


def get_calibration_rotation(rpy):
    """Convert roll, pitch, yaw to rotation matrix"""
    if not rpy or len(rpy) != 3:
        return np.eye(3)

    roll, pitch, yaw = rpy

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def project_points_to_screen(points_3d, K, R, img_width, img_height):
    """
    Project 3D points in car space to 2D screen coordinates

    points_3d: Nx3 array of (x, y, z) in car frame (x=forward, y=right, z=down)
    K: 3x3 camera intrinsic matrix
    R: 3x3 calibration rotation matrix

    Returns: Nx2 array of (u, v) pixel coordinates
    """
    if len(points_3d) == 0:
        return np.array([])

    # Transform to camera frame
    points_cam = (R @ points_3d.T).T

    # Filter out points behind camera (negative z in camera frame)
    valid_mask = points_cam[:, 2] > 0.1

    if not valid_mask.any():
        return np.array([])

    # Project to image plane using intrinsics
    points_homog = (K @ points_cam.T).T

    # Normalize by z coordinate
    points_2d = points_homog[:, :2] / points_homog[:, [2]]

    # Filter points outside image bounds
    valid_mask &= (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width)
    valid_mask &= (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)

    return points_2d[valid_mask].astype(np.int32) if valid_mask.any() else np.array([])


def draw_path(frame, path, K, R, color=(0, 255, 0), thickness=2):
    """Draw vehicle path overlay"""
    if not path or 'x' not in path:
        return

    # Convert path to 3D points
    points_3d = np.column_stack([path['x'], path['y'], path['z']])

    # Project to screen
    points_2d = project_points_to_screen(points_3d, K, R, frame.shape[1], frame.shape[0])

    if len(points_2d) > 1:
        # Draw path as polyline with gradient
        for i in range(len(points_2d) - 1):
            # Fade color with distance
            alpha = 1.0 - (i / len(points_2d))
            c = tuple(int(alpha * x) for x in color)
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[i+1]), c, thickness)


def draw_lane_line(frame, lane, K, R, color=(255, 255, 0), thickness=2):
    """Draw single lane line"""
    if not lane or 'x' not in lane:
        return

    # Convert to 3D points
    points_3d = np.column_stack([lane['x'], lane['y'], lane['z']])

    # Project to screen
    points_2d = project_points_to_screen(points_3d, K, R, frame.shape[1], frame.shape[0])

    if len(points_2d) > 1:
        # Scale alpha by probability
        alpha = lane.get('prob', 1.0)
        c = tuple(int(alpha * x) for x in color)

        # Draw as polyline
        cv2.polylines(frame, [points_2d], False, c, thickness, cv2.LINE_AA)


def draw_lead_car(frame, lead, K, R):
    """Draw lead car detection as bounding box"""
    if not lead or 'x' not in lead or len(lead['x']) == 0:
        return

    # Use first point (current position)
    x, y = lead['x'][0], lead['y'][0]
    prob = lead.get('prob', 1.0)

    if prob < 0.3:
        return

    # Approximate lead car as a box (1.8m wide, 1.5m tall, at position x,y)
    car_width = 1.8
    car_height = 1.5

    # Four corners of lead car (in car space)
    corners_3d = np.array([
        [x, y - car_width/2, 0],            # Bottom left
        [x, y + car_width/2, 0],            # Bottom right
        [x, y + car_width/2, -car_height],  # Top right
        [x, y - car_width/2, -car_height],  # Top left
    ])

    # Project to screen
    corners_2d = project_points_to_screen(corners_3d, K, R, frame.shape[1], frame.shape[0])

    if len(corners_2d) >= 4:
        # Draw bounding box
        color = (0, 0, 255) if prob > 0.7 else (0, 165, 255)  # Red for high confidence, orange otherwise
        cv2.polylines(frame, [corners_2d], True, color, 2, cv2.LINE_AA)

        # Draw distance label
        cv2.putText(frame, f"{x:.0f}m",
                   tuple(corners_2d[0]),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, color, 2)


def draw_model_overlay(frame, model_data, camera_name):
    """Draw all model predictions on frame"""
    if not model_data:
        return

    # Get camera intrinsics
    K = get_camera_intrinsics(camera_name)

    # Get calibration rotation
    cal = model_data.get('calibration', {})
    R = get_calibration_rotation(cal.get('rpy', [0, 0, 0]))

    # Draw vehicle path (green, thick)
    if 'path' in model_data:
        draw_path(frame, model_data['path'], K, R, color=(0, 200, 0), thickness=4)

    # Draw lane lines (yellow/blue)
    lane_colors = [
        (100, 100, 255),  # Left edge - red-ish
        (255, 255, 100),  # Left lane - cyan-ish
        (255, 255, 100),  # Right lane - cyan-ish
        (100, 100, 255),  # Right edge - red-ish
    ]

    for i, lane in enumerate(model_data.get('laneLines', [])):
        color = lane_colors[i % len(lane_colors)]
        draw_lane_line(frame, lane, K, R, color=color, thickness=2)

    # Draw road edges (orange, thick)
    for edge in model_data.get('roadEdges', []):
        draw_lane_line(frame, edge, K, R, color=(0, 165, 255), thickness=3)

    # Draw lead cars (red boxes)
    for lead in model_data.get('leads', []):
        draw_lead_car(frame, lead, K, R)


class StreamViewer:
    def __init__(self, base_url):
        self.base_url = base_url
        self.current_camera = 'road'
        self.cameras = ['road', 'driver', 'wide']
        self.camera_index = 0
        self.should_switch = False
        self.running = True
        self.show_overlay = True

    def get_url(self):
        return f"{self.base_url}?camera={self.current_camera}"

    def switch_camera(self, camera):
        if camera != self.current_camera and camera in self.cameras:
            self.current_camera = camera
            self.camera_index = self.cameras.index(camera)
            self.should_switch = True
            return True
        return False

    def next_camera(self):
        self.camera_index = (self.camera_index + 1) % len(self.cameras)
        self.current_camera = self.cameras[self.camera_index]
        self.should_switch = True


def stream_mjpeg_viewer(base_url):
    """View MJPEG stream with live camera switching and model overlays"""

    viewer = StreamViewer(base_url)

    # Create window
    window_name = "openpilot Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)

    print("Controls:")
    print("  q - Quit")
    print("  f - Toggle fullscreen")
    print("  s - Save screenshot")
    print("  o - Toggle model overlay")
    print("  1 - Switch to Road camera")
    print("  2 - Switch to Driver camera")
    print("  3 - Switch to Wide camera")
    print("  n - Next camera")
    print("  + - Increase window size")
    print("  - - Decrease window size")
    print("")

    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fullscreen = False

    try:
        while viewer.running:
            url = viewer.get_url()
            print(f"Connecting to {url}...")

            # Small delay to let previous connection close properly
            time.sleep(0.1)

            try:
                # Connect to stream
                response = requests.get(url, stream=True, timeout=5)

                if response.status_code != 200:
                    print(f"Error: Server returned status {response.status_code}")
                    time.sleep(1)
                    continue

                print(f"Connected! Streaming {viewer.current_camera} camera...")
                viewer.should_switch = False

                # Read MJPEG stream
                bytes_buffer = b''

                for chunk in response.iter_content(chunk_size=4096):
                    if viewer.should_switch:
                        print(f"Switching to {viewer.current_camera} camera...")
                        response.close()  # Explicitly close connection before switching
                        break

                    bytes_buffer += chunk

                    # Look for frame boundary
                    boundary_start = bytes_buffer.find(b'--frame')
                    if boundary_start == -1:
                        continue

                    # Look for headers and image data
                    header_end = bytes_buffer.find(b'\r\n\r\n', boundary_start)
                    if header_end == -1:
                        continue

                    # Extract headers
                    headers = bytes_buffer[boundary_start:header_end].decode('utf-8', errors='ignore')

                    # Parse Content-Length
                    content_length = None
                    for line in headers.split('\r\n'):
                        if line.startswith('Content-Length:'):
                            content_length = int(line.split(':')[1].strip())
                            break

                    if content_length is None:
                        continue

                    # Check if we have complete frame
                    image_start = header_end + 4
                    image_end = image_start + content_length

                    if len(bytes_buffer) < image_end:
                        continue

                    # Extract JPEG and model data
                    jpeg_data = bytes_buffer[image_start:image_end]

                    # Parse model data from headers
                    model_data = {}
                    for line in headers.split('\r\n'):
                        if line.startswith('X-Model-Data:'):
                            try:
                                json_str = line.split(':', 1)[1].strip()
                                model_data = json.loads(json_str)
                            except:
                                pass
                            break

                    # Move buffer forward
                    bytes_buffer = bytes_buffer[image_end:]

                    # Decode JPEG
                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Calculate FPS
                        frame_count += 1
                        current_time = time.time()

                        if current_time - last_fps_time >= 1.0:
                            fps = frame_count / (current_time - start_time)
                            cv2.setWindowTitle(window_name, f"openpilot [{viewer.current_camera.upper()}] - {fps:.1f} FPS")
                            last_fps_time = current_time

                        # Draw model overlay
                        if viewer.show_overlay and model_data:
                            draw_model_overlay(frame, model_data, viewer.current_camera)

                        # Add camera info overlay
                        overlay_y = 30
                        cv2.putText(frame, f"Camera: {viewer.current_camera.upper()}",
                                   (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        overlay_y += 30
                        if viewer.show_overlay:
                            cv2.putText(frame, "Overlay: ON (press 'o' to toggle)",
                                       (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            cv2.putText(frame, "Overlay: OFF (press 'o' to toggle)",
                                       (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                        # Controls hint at bottom
                        cv2.putText(frame, "1:Road | 2:Driver | 3:Wide | O:Overlay | Q:Quit",
                                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (200, 200, 200), 1)

                        # Display frame
                        cv2.imshow(window_name, frame)

                        # Handle keyboard input
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('q'):
                            print("Quit requested")
                            viewer.running = False
                            break
                        elif key == ord('f'):
                            fullscreen = not fullscreen
                            if fullscreen:
                                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            else:
                                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        elif key == ord('s'):
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"openpilot_{viewer.current_camera}_{timestamp}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"Screenshot saved: {filename}")
                        elif key == ord('o'):
                            viewer.show_overlay = not viewer.show_overlay
                            print(f"Model overlay: {'ON' if viewer.show_overlay else 'OFF'}")
                        elif key == ord('1'):
                            if viewer.switch_camera('road'):
                                print("Switching to ROAD camera...")
                        elif key == ord('2'):
                            if viewer.switch_camera('driver'):
                                print("Switching to DRIVER camera...")
                        elif key == ord('3'):
                            if viewer.switch_camera('wide'):
                                print("Switching to WIDE camera...")
                        elif key == ord('n'):
                            viewer.next_camera()
                            print(f"Switching to {viewer.current_camera.upper()} camera...")
                        elif key == ord('+') or key == ord('='):
                            w, h = cv2.getWindowImageRect(window_name)[2:]
                            cv2.resizeWindow(window_name, int(w * 1.2), int(h * 1.2))
                        elif key == ord('-') or key == ord('_'):
                            w, h = cv2.getWindowImageRect(window_name)[2:]
                            cv2.resizeWindow(window_name, int(w * 0.8), int(h * 0.8))

            except requests.exceptions.ConnectionError:
                print(f"Connection error. Retrying in 2 seconds...")
                time.sleep(2)
            except requests.exceptions.Timeout:
                print(f"Connection timeout. Retrying...")
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping...")
                viewer.running = False
                break

    finally:
        cv2.destroyAllWindows()
        print("Viewer closed")


def main():
    parser = argparse.ArgumentParser(description="View openpilot camera + model overlay stream")
    parser.add_argument('device_ip', nargs='?', default='192.168.43.1',
                        help='IP address of the openpilot device (default: 192.168.43.1)')
    parser.add_argument('--port', type=int, default=8000, help='HTTP port (default: 8000)')
    args = parser.parse_args()

    base_url = f"http://{args.device_ip}:{args.port}"

    print("=" * 60)
    print("openpilot Advanced Stream Viewer")
    print("=" * 60)
    print(f"Server: {base_url}")
    print(f"Features: Live camera switching + Model overlays")
    print("=" * 60)

    stream_mjpeg_viewer(base_url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
