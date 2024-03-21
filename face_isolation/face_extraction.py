import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

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
            face = cv2.resize(face, self.reshape_size)
            return face
        except Exception as e:
            print(f"failed to extract face because of: '{e}'")
            return None

if __name__ == '__main__':
    mtcnn = MTCNN()
    fcd = FaceExtraction(mtcnn, reshape_size=(128,128))
    img = cv2.imread('./dataset/real_and_fake_face/fake/easy_1_1110.jpg')
    face = fcd.extract_face(img)
    if face is not None:
        cv2.imshow('Extracted Face', face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No face detected in the image.")

        