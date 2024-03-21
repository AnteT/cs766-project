## CS 766 - Project Repo
#### Udhbhav Gupta & Ante Tonkovic-Capin

---

Aritifcal Facial Detection, Separating Real Faces from the Artificially Generated 

#### Project Structure

```shell
Project
   ├── README.MD
   ├── dataset
   │   └── real_and_fake_face
   │       ├── fake
   │       │   └── images...
   │       └── real
   │          └── images...
   │
   ├── model.py
   ├── project-docs
   │   ├── Proposal.MD
   │   └── Proposal.pdf
   │
   ├── face_isolation
   │   ├── __init__.py
   │   ├── face_extraction.py
   │   └── face_isolation.py
   │
   ├── requirements.txt
   └── train_loop.py
```

#### Step 1: Extract Face from Images
Perhaps using one of the models from https://github.com/timesler/facenet-pytorch to extract the face, or initial structure in `./face_isolation/face_isolation`

#### Step 2: Train Fake or Real Face Classifier
Basic model setup in `./model.py`, still not getting test accuracy over ~55% so maybe need to try a new approach
