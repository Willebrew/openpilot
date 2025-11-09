#!/usr/bin/env python3
"""
Fast H.264 Camera Stream Server for openpilot
Uses the hardware-encoded video stream - much faster than JPEG encoding!

Usage: python3 stream_server.py [--port PORT] [--camera CAMERA]
"""

import argparse
import socket
import struct
import subprocess
import time

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.system.hardware import PC
from openpilot.system.manager.process_config import managed_processes


ENCODE_SOCKETS = {
    "road": "roadEncodeData",
    "driver": "driverEncodeData",
    "wide": "wideRoadEncodeData",
}


def send_packet(sock, data):
    """Send a packet with length header"""
    sock.sendall(struct.pack('>I', len(data)))
    sock.sendall(data)


def stream_encoded_video(camera_name, port):
    """Stream hardware-encoded H.264 video over TCP"""
    print(f"Starting H.264 stream server for '{camera_name}' camera on port {port}")

    # Check if encoderd is already running
    encoderd_running = False
    try:
        subprocess.check_call(["pgrep", "encoderd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("encoderd already running")
        encoderd_running = True
    except subprocess.CalledProcessError:
        print("encoderd not running, starting it...")
        if not PC:
            try:
                # Start camerad first (encoderd depends on it)
                try:
                    subprocess.check_call(["pgrep", "camerad"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print("camerad already running")
                except subprocess.CalledProcessError:
                    managed_processes['camerad'].start()
                    print("camerad started")
                    time.sleep(2)

                managed_processes['encoderd'].start()
                print("encoderd started successfully")
                time.sleep(2)  # Give encoderd time to initialize
            except Exception as e:
                print(f"Warning: Could not start encoderd: {e}")
                print("Attempting to continue anyway...")
        else:
            print("Running on PC, encoderd may need to be started manually")

    # Setup messaging
    sock_name = ENCODE_SOCKETS[camera_name]
    print(f"Subscribing to {sock_name}...")
    sm = messaging.SubMaster([sock_name])

    # Wait for first frame
    print("Waiting for encoded frames...")
    max_wait = 30
    wait_start = time.time()
    while sm[sock_name].idx.frameId == 0:
        sm.update(100)
        if time.time() - wait_start > max_wait:
            print(f"Warning: No frames after {max_wait}s")
            break

    print(f"Receiving frames from {sock_name}")

    # Setup TCP server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', port))
    server_sock.listen(1)
    server_sock.settimeout(1.0)

    print(f"Server listening on 0.0.0.0:{port}")
    print("Ready for client connections!")
    print("\nTo view on your computer, run:")
    print(f"  python3 stream_client.py <device_ip> --port {port}")

    try:
        while True:
            # Accept client connection
            try:
                client_sock, addr = server_sock.accept()
                print(f"\nClient connected from {addr}")
            except socket.timeout:
                continue

            try:
                frame_count = 0
                start_time = time.time()
                last_fps_time = start_time
                sent_header = False

                while True:
                    sm.update(0)  # Non-blocking
                    msg = sm[sock_name]

                    if msg.idx.frameId == 0:
                        time.sleep(0.01)
                        continue

                    # Send header on first frame or keyframe
                    if not sent_header and len(msg.header) > 0:
                        send_packet(client_sock, bytes(msg.header))
                        sent_header = True
                        print("Sent H.264 header")

                    # Send frame data
                    if len(msg.data) > 0:
                        send_packet(client_sock, bytes(msg.data))
                        frame_count += 1

                        # Print stats
                        current_time = time.time()
                        if current_time - last_fps_time >= 1.0:
                            fps = frame_count / (current_time - start_time)
                            print(f"Streaming at {fps:.1f} FPS | Packet size: {len(msg.data)/1024:.1f} KB")
                            last_fps_time = current_time

            except (BrokenPipeError, ConnectionResetError):
                print("Client disconnected")
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            finally:
                client_sock.close()

    finally:
        server_sock.close()
        # Stop encoderd if we started it
        if not encoderd_running and not PC:
            print("Stopping encoderd...")
            try:
                managed_processes['encoderd'].stop()
            except Exception as e:
                print(f"Warning: Could not stop encoderd: {e}")
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Stream openpilot H.264 video over TCP")
    parser.add_argument('--port', type=int, default=5555, help='TCP port to listen on (default: 5555)')
    parser.add_argument('--camera', choices=['road', 'driver', 'wide'], default='road',
                        help='Camera to stream (default: road)')
    args = parser.parse_args()

    print("=" * 60)
    print("openpilot H.264 Stream Server (FAST)")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Port: {args.port}")
    print(f"Encoding: Hardware H.264 (20 FPS)")
    print("=" * 60)

    stream_encoded_video(args.camera, args.port)


if __name__ == "__main__":
    main()
