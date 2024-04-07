import sys
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from PIL import Image

class FaceExtraction(object):
    """Face detector class to isolate faces or features if needed from images."""
    def __init__(self, reshape_size:tuple[int]=(128,128)):
        self.mtcnn = MTCNN()
        self.reshape_size = reshape_size

    def extract_face(self, frame):
        """Extract face from the frame and return as resized image if needed."""
        try:
            boxes, _ = self.mtcnn.detect(frame)
            if boxes is None:
                return None
            box = boxes[0]
            box = box.astype(np.uint32)
            face = frame[box[1]:box[3], box[0]:box[2]]
            face = cv2.resize(face, self.reshape_size)
            return face
        except Exception as e:
            print(f"failed to extract face because of: '{e}'")
            raise ValueError

class FaceExtractionTransform:
    """Face Extraction and Transformation to apply to torch dataloader class to extract only relevant facial features when training the model."""
    def __init__(self, reshape_size):
        self.mtcnn = FaceExtraction()
        self.reshape_size = reshape_size

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        try:
            face = self.mtcnn.extract_face(img)
        except ValueError:
            return cv2.resize(img, self.reshape_size)
        if face is not None:
            # face = cv2.resize(face, self.reshape_size, interpolation=cv2.INTER_NEAREST)
            face = cv2.resize(face, self.reshape_size)
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Convert to torch tensor
            face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1) / 255.0
            return face
        else:
            # Return original image if face is not detected
            return cv2.resize(img, self.reshape_size)