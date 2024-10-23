"""
This script performs live inference using a pre-trained ResNet-18 model for binary
classification of parking vehicles.

Modules:
    - torch: PyTorch library for deep learning.
    - torchvision: PyTorch library for vision-related tasks.
    - cv2: OpenCV library for real-time computer vision.

Functions:
    - predict_object(frame, threshold=0.5): Predicts the presence of a vehicle in the given frame.
"""
import os
import time
import torch
from torchvision.models import resnet18
import cv2
from model_utils import get_transform
import numpy as np
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.params import Params
import cereal.messaging as messaging

# Define the image transformation pipeline
transform = get_transform()

# Load ResNet-18
model = resnet18()
# Modify the last layer for binary classification
model.fc = torch.nn.Linear(model.fc.in_features, 1)

# Load the model weights
model.load_state_dict(torch.load('parking_vehicle_model.pth'))
model.eval()

def predict_object(image_frame, threshold=0.5):
    """
    Predicts whether an object in the frame is a campus parking service vehicle.
    :param image_frame:
    :param threshold:
    :return:
    """
    image_tensor = transform(image_frame).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        live_prediction = torch.sigmoid(output).item()

    return live_prediction, live_prediction > threshold

def main():
    # Setup Vision IPC client
    cl_context = CLContext()
    while True:
        available_streams = VisionIpcClient.available_streams("camerad", block=False)
        if available_streams:
            main_wide_camera = VisionStreamType.VISION_STREAM_DRIVER in available_streams
            break

    vipc_client_main_stream = VisionStreamType.VISION_STREAM_DRIVER
    vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
    vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, False, cl_context)

    while not vipc_client_main.connect(False):
        time.sleep(0.1)
    while not vipc_client_extra.connect(False):
        time.sleep(0.1)

    model_transform_main = np.zeros((3, 3), dtype=np.float32)
    model_transform_extra = np.zeros((3, 3), dtype=np.float32)

    while True:
        buf_main = vipc_client_main.recv()
        if buf_main is None:
            print("Failed to grab frame from main camera")
            continue

        buf_extra = vipc_client_extra.recv()
        if buf_extra is None:
            print("Failed to grab frame from extra camera")
            continue

        frame = buf_main.image  # Assuming buf_main.image holds the frame data

        prediction, detected = predict_object(frame, threshold=0.5)

        # Log the prediction and detection status
        if detected:
            print(f"CAMPUS PARKING SERVICES DETECTED: {prediction:.2f}")
        else:
            print(f"NO THREATS DETECTED: {prediction:.2f}")

    vipc_client_main.close()
    vipc_client_extra.close()

if __name__ == "__main__":
    main()
