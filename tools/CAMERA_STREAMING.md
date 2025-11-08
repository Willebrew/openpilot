# openpilot Camera Streaming

This directory contains tools to stream live camera feeds from your openpilot device to your computer over SSH/network.

## Overview

The camera streaming system consists of two scripts:

1. **camera_stream_server.py** - Runs on the openpilot device
2. **camera_stream_client.py** - Runs on your computer

## How It Works

### Hardware Integration

openpilot uses the Qualcomm Spectra ISP (Image Signal Processor) to capture frames from three cameras:

- **Road Camera** (8.0mm focal length) - Main forward-facing camera
- **Wide Road Camera** (1.71mm focal length) - Wide-angle forward camera
- **Driver Camera** (1.71mm focal length) - Driver monitoring camera

The cameras are accessed through:
- **VisionIPC**: Shared memory interface for raw YUV frames (low-latency, local only)
- **cereal messaging**: Cap'n Proto messages for camera state (network-capable)

### Streaming Architecture

```
Device (openpilot):
  camerad → VisionIPC → camera_stream_server.py
                            ↓
                          YUV → RGB → JPEG
                            ↓
                        TCP Socket (port 5555)
                            ↓
Computer:                   ↓
  camera_stream_client.py ← TCP Socket
         ↓
    JPEG decode → OpenCV display
```

## Prerequisites

### On the Device (openpilot)
- openpilot must be installed and camerad must be running
- No additional dependencies needed (uses openpilot's environment)

### On Your Computer
- Python 3.7+
- Install dependencies:
  ```bash
  pip install opencv-python numpy pillow
  ```

## Usage

### Step 1: SSH into your device

```bash
ssh comma@<device_ip>
```

Default password is typically shown on the device screen or in openpilot settings.

### Step 2: Start the server on the device

Navigate to the openpilot directory and run:

```bash
cd /data/openpilot  # or wherever openpilot is installed
python3 tools/camera_stream_server.py
```

**Options:**
- `--port PORT` - TCP port to listen on (default: 5555)
- `--camera CAMERA` - Camera to stream: `road`, `driver`, or `wide` (default: `road`)
- `--quality QUALITY` - JPEG quality 1-100 (default: 85)

**Examples:**
```bash
# Stream road camera on default port 5555
python3 tools/camera_stream_server.py

# Stream driver camera on port 6000 with high quality
python3 tools/camera_stream_server.py --camera driver --port 6000 --quality 95

# Stream wide camera with lower quality for faster streaming
python3 tools/camera_stream_server.py --camera wide --quality 70
```

### Step 3: Run the client on your computer

On your computer, run:

```bash
python3 camera_stream_client.py <device_ip>
```

**Options:**
- `--port PORT` - TCP port to connect to (default: 5555)

**Examples:**
```bash
# Connect to device at 192.168.1.100
python3 camera_stream_client.py 192.168.1.100

# Connect to custom port
python3 camera_stream_client.py 192.168.1.100 --port 6000
```

### Client Controls

When using OpenCV display mode:
- `q` - Quit the viewer
- `s` - Save current frame as screenshot (saved to current directory)
- `f` - Toggle FPS display on/off

## Camera Details

### Road Camera
- **Purpose**: Main forward-facing camera for driving model
- **Resolution**: Typically 1164x874 or similar (varies by device)
- **Focal Length**: 8.0mm
- **Frame Rate**: 20 Hz
- **Use Case**: Best for viewing the road ahead

### Driver Camera
- **Purpose**: Driver monitoring (face detection, alertness)
- **Resolution**: Typically 1164x874 or similar
- **Focal Length**: 1.71mm (wide angle)
- **Frame Rate**: 20 Hz
- **Use Case**: View the driver cabin

### Wide Road Camera
- **Purpose**: Wide-angle view for better peripheral vision
- **Resolution**: Typically 1164x874 or similar
- **Focal Length**: 1.71mm (wide angle)
- **Frame Rate**: 20 Hz
- **Use Case**: Wide-angle road view

## Performance Tips

### For Best Performance:
1. **Use OpenCV**: Install `opencv-python` on your computer for hardware-accelerated display
2. **Adjust Quality**: Lower JPEG quality (60-75) reduces bandwidth and latency
3. **Local Network**: Connect device and computer to same WiFi network
4. **Wired Connection**: Use ethernet/USB tethering for lowest latency

### Typical Performance:
- **FPS**: 15-20 FPS (depends on network)
- **Latency**: 100-300ms (local network)
- **Bandwidth**: ~500 KB/frame @ 85% quality = ~10 MB/s @ 20 FPS

### Troubleshooting:

**Server won't start:**
- Check if camerad is running: `ps aux | grep camerad`
- Make sure you're running as the correct user (comma)
- Check if port is already in use: `lsof -i :5555`

**Client can't connect:**
- Verify device IP: `ip addr show` on device
- Check firewall settings
- Ensure port matches on both server and client
- Test connectivity: `ping <device_ip>`

**Low FPS / Lag:**
- Reduce JPEG quality: `--quality 60`
- Check network bandwidth
- Close other network-intensive applications
- Switch to wired connection if possible

**No frames / Black screen:**
- Wait 4-5 seconds for camera auto-exposure to adjust
- Check if camerad is actually capturing: `cereal.log roadCameraState` on device
- Try a different camera: `--camera driver` or `--camera wide`

## Technical Details

### Frame Format
- **Capture**: YUV420 NV12 (from VisionIPC)
- **Conversion**: YUV → RGB using ITU-R BT.601 color space
- **Transport**: JPEG compressed
- **Protocol**: TCP with length-prefixed frames (4-byte header + data)

### Latency Breakdown
1. Camera capture: ~50ms (1 frame @ 20Hz)
2. YUV→RGB conversion: ~5-10ms
3. JPEG encoding: ~10-20ms
4. Network transmission: ~50-200ms (depends on network)
5. JPEG decoding: ~5-10ms
6. Display: ~16ms (60 FPS display)

**Total**: ~136-316ms typical latency

### Code References
- Camera access: `system/camerad/snapshot.py:53-73`
- VisionIPC client: `msgq/visionipc/`
- Camera services: `cereal/services.py:60-66`
- Frame conversion: Based on `system/camerad/snapshot.py:29-50`

## Alternative Methods

### Using compressed_vipc.py (existing tool)
For more advanced streaming with hardware video encoding:

```bash
# On device: Enable cereal messaging bridge and run encoderd
# On computer:
python3 tools/camerastream/compressed_vipc.py <device_ip>
```

This uses HEVC hardware encoding for better compression but requires more setup.

### Using cereal messaging directly
For lowest overhead (but requires openpilot environment on receiving end):

```python
import cereal.messaging as messaging
sm = messaging.SubMaster(['roadCameraState'], addr=<device_ip>)
# Process camera state messages
```

## Safety Notice

**Do not use this tool while driving.** This is for development, debugging, and testing purposes only. Viewing camera streams should only be done when the vehicle is parked safely.

## License

Part of the openpilot project. See main LICENSE file.
