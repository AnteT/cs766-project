import sys
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from PIL import Image

class FaceExtractionTransform:
    """Face Extraction and Transformation to apply to torch dataloader class to extract only relevant facial features when training the model."""
    def __init__(self, mtcnn, reshape_size):
        self.mtcnn = mtcnn
        self.reshape_size = reshape_size

    def __call__(self, img):
        face = self.mtcnn.extract_face(img)
        if face is not None:
            # Resize the face image
            face = cv2.resize(face, self.reshape_size, interpolation=cv2.INTER_NEAREST)
            # Convert BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Convert to torch tensor
            face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1) / 255.0
            return face
        else:
            # Return None if face is not detected
            return None

class FaceExtraction(object):
    """Face detector class to isolate faces or features if needed from images."""
    def __init__(self, mtcnn, reshape_size:tuple[int]=(128,128)):
        self.mtcnn = mtcnn
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
            face = cv2.resize(face, self.reshape_size, interpolation=cv2.INTER_NEAREST)
            return face
        except Exception as e:
            print(f"failed to extract face because of: '{e}'")
            return None

def test_extract_face_from_image(img_path:str) -> None:
    """Testing feature extraction from image path."""
    mtcnn = MTCNN()
    fcd = FaceExtraction(mtcnn, reshape_size=(128, 128))
    img = cv2.imread(img_path)
    face = fcd.extract_face(img)
    if face is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Extracted Face')
        axes[1].axis('off')
        plt.show()
    else:
        print("No face detected in the image.")

if __name__ == '__main__':
    test_img = 'C:/Users/212765834/Desktop/Personal Repos/CS766 - Computer Vision/CS766 - CV Final Project/dataset/140k/real_vs_fake/real-vs-fake/train/fake/0A0IAK9X2W.jpg'
    test_extract_face_from_image(test_img)
    sys.exit()

    mtcnn = MTCNN()
    fcd = FaceExtraction(mtcnn, reshape_size=(128,128))
    img = cv2.imread(test_img)
    face = fcd.extract_face(img)
    if face is not None:
        cv2.imshow('Extracted Face', face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected in the image.")

        