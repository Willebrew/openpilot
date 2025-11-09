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


# Camera intrinsics (from openpilot/common/transformations/camera.py)
# These are the exact values used by openpilot
CAMERA_CONFIGS = {
    'road': {
        'focal_length': 2648.0,  # pixels (not mm!)
        'width': 1928,
        'height': 1208,
    },
    'wide': {
        'focal_length': 567.0,  # pixels
        'width': 1928,
        'height': 1208,
    },
    'driver': {
        'focal_length': 567.0,  # pixels
        'width': 1928,
        'height': 1208,
    }
}

# Frame transformation matrix (from openpilot/common/transformations/camera.py)
# device frame: x->forward, y->right, z->down
# view frame: x->right, y->down, z->forward
VIEW_FRAME_FROM_DEVICE_FRAME = np.array([
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.]
])


def get_camera_intrinsics(camera_name):
    """Get camera intrinsic matrix (same as openpilot)"""
    cfg = CAMERA_CONFIGS[camera_name]
    focal_length = cfg['focal_length']
    width = cfg['width']
    height = cfg['height']

    # Intrinsic matrix K (camera_frame_from_view_frame)
    K = np.array([
        [focal_length, 0.0, width / 2.0],
        [0.0, focal_length, height / 2.0],
        [0.0, 0.0, 1.0]
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


def calculate_transform(camera_name, rpy_calib, img_width, img_height):
    """
    Calculate the full transformation matrix from device frame to screen coordinates.
    This replicates openpilot's augmented_road_view.py transformation pipeline.

    Returns: 3x3 transformation matrix
    """
    # Get camera intrinsics
    intrinsic = get_camera_intrinsics(camera_name)
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # Get calibration rotation (device_from_calib)
    device_from_calib = get_calibration_rotation(rpy_calib)

    # Calculate view_from_calib (from augmented_road_view.py line 154)
    view_from_calib = VIEW_FRAME_FROM_DEVICE_FRAME @ device_from_calib

    # Calculate calib_transform (from augmented_road_view.py line 180)
    calib_transform = intrinsic @ view_from_calib

    # Calculate zoom (wide camera uses 2.0x zoom, road uses 1.1x)
    zoom = 2.0 if camera_name == 'wide' else 1.1

    # For simplicity, we don't calculate vanishing point offsets here
    # (would need to project infinity point like augmented_road_view.py does)
    # Just use centered view
    x_offset = 0.0
    y_offset = 0.0

    # Calculate video transform (from augmented_road_view.py lines 211-215)
    x = 0  # Content rect origin (no border in our viewer)
    y = 0
    w = img_width
    h = img_height

    video_transform = np.array([
        [zoom, 0.0, (w / 2 + x - x_offset) - (cx * zoom)],
        [0.0, zoom, (h / 2 + y - y_offset) - (cy * zoom)],
        [0.0, 0.0, 1.0]
    ])

    # Final transform (from augmented_road_view.py line 216)
    final_transform = video_transform @ calib_transform

    return final_transform


def project_points_to_screen(points_3d, transform, img_width, img_height):
    """
    Project 3D points in device frame to 2D screen coordinates using openpilot's exact method.
    This replicates model_renderer.py _map_to_screen() function.

    points_3d: Nx3 array of (x, y, z) in device frame (x=forward, y=right, z=down)
    transform: 3x3 transformation matrix from calculate_transform()

    Returns: Nx2 array of (u, v) pixel coordinates
    """
    if len(points_3d) == 0:
        return np.array([])

    # Transform points (model_renderer.py line 322)
    # pt = self._car_space_transform @ input_pt
    points_transformed = (transform @ points_3d.T).T  # Shape: Nx3

    # Filter out points with invalid z (behind camera or too close)
    valid_mask = np.abs(points_transformed[:, 2]) >= 1e-6

    if not valid_mask.any():
        return np.array([])

    points_transformed = points_transformed[valid_mask]

    # Perspective division (model_renderer.py line 327)
    # x, y = pt[0] / pt[2], pt[1] / pt[2]
    points_2d = points_transformed[:, :2] / points_transformed[:, [2]]

    # Filter points outside reasonable screen bounds (with margin like openpilot does)
    margin = 500  # CLIP_MARGIN from model_renderer.py
    valid_screen = (
        (points_2d[:, 0] >= -margin) & (points_2d[:, 0] < img_width + margin) &
        (points_2d[:, 1] >= -margin) & (points_2d[:, 1] < img_height + margin)
    )

    if not valid_screen.any():
        return np.array([])

    return points_2d[valid_screen].astype(np.int32)


def draw_path(frame, path, transform, color=(0, 255, 0), thickness=2):
    """Draw vehicle path overlay"""
    if not path or 'x' not in path:
        return

    # Convert path to 3D points in device frame
    points_3d = np.column_stack([path['x'], path['y'], path['z']])

    # Project to screen
    points_2d = project_points_to_screen(points_3d, transform, frame.shape[1], frame.shape[0])

    if len(points_2d) > 1:
        # Draw path as polyline with gradient
        for i in range(len(points_2d) - 1):
            # Fade color with distance
            alpha = 1.0 - (i / len(points_2d))
            c = tuple(int(alpha * x) for x in color)
            cv2.line(frame, tuple(points_2d[i]), tuple(points_2d[i+1]), c, thickness)


def draw_lane_line(frame, lane, transform, color=(255, 255, 0), thickness=2):
    """Draw single lane line"""
    if not lane or 'x' not in lane:
        return

    # Convert to 3D points in device frame
    points_3d = np.column_stack([lane['x'], lane['y'], lane['z']])

    # Project to screen
    points_2d = project_points_to_screen(points_3d, transform, frame.shape[1], frame.shape[0])

    if len(points_2d) > 1:
        # Scale alpha by probability
        alpha = lane.get('prob', 1.0)
        c = tuple(int(alpha * x) for x in color)

        # Draw as polyline
        cv2.polylines(frame, [points_2d], False, c, thickness, cv2.LINE_AA)


def draw_lead_car(frame, lead, transform):
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

    # Four corners of lead car (in device frame: x=forward, y=right, z=down)
    corners_3d = np.array([
        [x, y - car_width/2, 0],            # Bottom left
        [x, y + car_width/2, 0],            # Bottom right
        [x, y + car_width/2, -car_height],  # Top right
        [x, y - car_width/2, -car_height],  # Top left
    ])

    # Project to screen
    corners_2d = project_points_to_screen(corners_3d, transform, frame.shape[1], frame.shape[0])

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
    """Draw all model predictions on frame using openpilot's exact transformation"""
    if not model_data:
        return

    # Get calibration data
    cal = model_data.get('calibration', {})
    rpy_calib = cal.get('rpy', [0, 0, 0])

    # Calculate the full transformation matrix (same as openpilot)
    transform = calculate_transform(camera_name, rpy_calib, frame.shape[1], frame.shape[0])

    # Draw vehicle path (green, thick)
    if 'path' in model_data:
        draw_path(frame, model_data['path'], transform, color=(0, 200, 0), thickness=4)

    # Draw lane lines (yellow/blue)
    lane_colors = [
        (100, 100, 255),  # Left edge - red-ish
        (255, 255, 100),  # Left lane - cyan-ish
        (255, 255, 100),  # Right lane - cyan-ish
        (100, 100, 255),  # Right edge - red-ish
    ]

    for i, lane in enumerate(model_data.get('laneLines', [])):
        color = lane_colors[i % len(lane_colors)]
        draw_lane_line(frame, lane, transform, color=color, thickness=2)

    # Draw road edges (orange, thick)
    for edge in model_data.get('roadEdges', []):
        draw_lane_line(frame, edge, transform, color=(0, 165, 255), thickness=3)

    # Draw lead cars (red boxes)
    for lead in model_data.get('leads', []):
        draw_lead_car(frame, lead, transform)


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
