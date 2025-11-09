#!/usr/bin/env python3
"""
Fast H.264 Camera Stream Client for openpilot
Receives hardware-encoded H.264 stream and plays with ffplay

Usage: python3 stream_client.py <device_ip> [--port PORT]

Requirements:
    - ffplay (from ffmpeg package)

Install ffmpeg:
    macOS: brew install ffmpeg
    Linux: sudo apt install ffmpeg
"""

import argparse
import socket
import struct
import subprocess
import sys


def recv_exact(sock, n):
    """Receive exactly n bytes from socket"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def recv_packet(sock):
    """Receive a packet with length header"""
    raw_len = recv_exact(sock, 4)
    if not raw_len:
        return None
    packet_len = struct.unpack('>I', raw_len)[0]
    packet_data = recv_exact(sock, packet_len)
    return packet_data


def stream_to_ffplay(sock, device_ip, port):
    """Receive H.264 stream and pipe to ffplay for display"""
    # Check if ffplay is available
    try:
        subprocess.run(["ffplay", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffplay not found!")
        print("\nPlease install ffmpeg:")
        print("  macOS:  brew install ffmpeg")
        print("  Linux:  sudo apt install ffmpeg")
        return

    # Start ffplay
    print(f"Starting ffplay to display stream from {device_ip}:{port}...")
    print("\nControls:")
    print("  q - Quit")
    print("  f - Toggle fullscreen")
    print("  p or Space - Pause/unpause")
    print("  s - Step to next frame (when paused)")
    print("\n" + "="*60)

    # Use ffmpeg to convert the stream to a format ffplay can handle
    # First, pipe through ffmpeg to decode and re-encode
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "h264",
        "-i", "pipe:0",
        "-c:v", "copy",            # Just copy, don't re-encode
        "-f", "mpegts",            # Output as MPEG-TS (better for streaming)
        "-tune", "zerolatency",
        "-",                       # Output to stdout
    ]

    ffplay_cmd = [
        "ffplay",
        "-f", "mpegts",            # Input is MPEG-TS
        "-fflags", "nobuffer",     # Minimize buffering
        "-flags", "low_delay",     # Low latency mode
        "-framedrop",              # Drop frames if behind
        "-i", "pipe:0",            # Read from stdin
        "-window_title", f"openpilot {device_ip}",
        "-loglevel", "warning",    # Show warnings
    ]

    try:
        # Start ffmpeg to mux the stream
        ffmpeg = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # Start ffplay to display the muxed stream
        ffplay = subprocess.Popen(
            ffplay_cmd,
            stdin=ffmpeg.stdout,
            stdout=subprocess.DEVNULL,
        )

        packet_count = 0
        try:
            while True:
                # Receive packet
                packet = recv_packet(sock)
                if packet is None:
                    print("\nConnection closed by server")
                    break

                # Send to ffmpeg
                try:
                    ffmpeg.stdin.write(packet)
                    ffmpeg.stdin.flush()
                    packet_count += 1

                    if packet_count % 100 == 0:
                        print(f"Received {packet_count} packets...", end='\r')

                except BrokenPipeError:
                    print("\nPipeline closed")
                    break

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            try:
                ffmpeg.stdin.close()
            except:
                pass
            ffmpeg.terminate()
            ffplay.terminate()
            ffmpeg.wait(timeout=2)
            ffplay.wait(timeout=2)

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="View openpilot H.264 stream with ffplay")
    parser.add_argument('device_ip', help='IP address of the openpilot device')
    parser.add_argument('--port', type=int, default=5555, help='TCP port (default: 5555)')
    args = parser.parse_args()

    print("=" * 60)
    print("openpilot H.264 Stream Client")
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
        print(f"  python3 stream_server.py --port {args.port}")
        return 1
    except socket.gaierror:
        print(f"Error: Invalid IP address: {args.device_ip}")
        return 1

    try:
        stream_to_ffplay(sock, args.device_ip, args.port)
    finally:
        sock.close()
        print("Disconnected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
