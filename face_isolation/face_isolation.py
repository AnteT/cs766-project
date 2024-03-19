# adapted from tutorial at https://medium.com/@iselagradilla94/how-to-build-a-face-detection-application-using-pytorch-and-opencv-d46b0866e4d6
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

class FaceIsolation(object):
    """Face detector class to isolate faces or features if needed from images."""
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        """Draw landmarks and boxes for each face detected."""
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])
                ld = ld.astype(np.uint32)

                # Draw rectangle on frame
                cv2.rectangle(frame,pt1,pt2,(0, 0, 255),thickness=2)

                # Show probability
                cv2.putText(frame, str(
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)

        except Exception as e:
            print(f"failed to draw because of: '{e}'")
        return frame

    def run(self):
        """Run the FaceIsolation and draw landmarks and boxes around detected faces."""
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            try:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                boxes = boxes.astype(np.uint32)
                self._draw(frame, boxes, probs, landmarks)
            except Exception as e:
                print(f"no draw due to '{e}'")
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    mtcnn = MTCNN()
    fcd = FaceIsolation(mtcnn)
    fcd.run()