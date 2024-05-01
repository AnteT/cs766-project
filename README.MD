## Artificial Facial Detection & Classification: Separating Real Faces from the Artificially Generated
#### Ante Tonkovic-Capin & Udhbhav Gupta

---
### Project Structure
This repository is a part of a broader submission for Computer Vision 766 at the University of Wisconsin-Madison for Spring 2024 semester. It represents the source code portion and is a companion repository for the related [website repository](https://github.com/AnteT/ffd-app) which contains the primary project details. The website is available [here](https://antet.github.io/ffd-app), while the rest of the project source is structured as follows:

```text
Project
  ├── README.MD
  ├── requirements.txt
  ├── face_isolation
  │   ├── __init__.py
  │   ├── face_extraction.py
  │   ├── face_extraction_transform.py
  │   └── face_isolation.py
  ├── models
  │   ├── ffd-s10000-e50-lr3e-04.pt
  │   ├── ffd-s5000-e20-lr3e-04.pt
  │   ├── ffd-s5000-e50-lr3e-04.pt
  │   ├── ffd-s5000-e80-lr1e-04.pt
  │   ├── ffd-s5000-e80-lr2e-04.pt
  │   └── graphs
  │       ├── ffd-onnx-horizontal-full.png
  │       ├── ffd-onnx-horizontal.png
  │       └── ffd-onnx-vertical.png
  ├── project-docs
  │   ├── Midterm Project Report.docx
  │   ├── Midterm Project Report.pdf
  │   ├── Midterm-Report-Guide.MD
  │   ├── Proposal.MD
  │   └── Proposal.pdf
  ├── model.py
  ├── model_inference.py
  ├── preprocess_img_mtcnn.py
  ├── ffx_plus_ffd_intuition.py
  ├── test_model.py
  ├── train_loop.py
  ├── train_val_test_loop.py
  └── train_val_test_loop_face_extract.py

4 directories, 27 files
```

Dataset containing 140k images used as primary training dataset is [available here](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces), while secondary dataset containing Photoshopped images can be found [here](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection). To reproduce results or train a new model, follow the instructions below.

---

### Run Training
Install the inference dependencies with `pip install -r requirements.txt` and the datasets desired. To reproduce project results, download and extract all 140,000 images from this kaggle dataset, then run the primary training, validation and testing script `train_val_test_loop.py` for training the model without a facial extraction phase, and run `train_val_test_loop.py` for training the model with the facial extraction phase before the CNN classifier phase. Confirm against full test set by running:

```bash
$ python ./model_inference.py
```

Resultant test accuracies and loss should reflect results referenced [here](https://antet.github.io/ffd-app/#results) 

---

### Run Demonstration
Proceed to the [companion repository](https://github.com/AnteT/ffd-app) which was made for using the pretrained model with the best results as opposed to this repository, which is made for retraining/reproducing the full model using all the images from the various datasets. Then follow instructions there or those available below:

Install the inference dependencies with `pip install -r requirements.txt`, then run the primary inference script `run_inference.py` using either the provided example images or your own. To run inference on your own images:

```bash
$ python ./run_inference.py [-d] "image1.png" ["image2.png"] ...
```

Multiple images can be provided and at least one is required for usage. Provide `-d, --display` flag to display image with the result. To reproduce the [demo](https://antet.github.io/ffd-app/#demonstration) referenced on the project website, use the `run_demo()` function or run:

```bash
$ python ./run_inference.py "./real-face.jpg" "./fake-face.jpg" --display 
```

Ensure the pretrained model `ffd.pt` and sample images, `real-face.jpg` and `fake-face.jpg` were correctly downloaded and are available in the project directory before running inference.

---

### Source Code Repository

The full project repository, source code, training results, prior models, training datasets, project proposals and midterm report are all available in the [project source repository](https://github.com/AnteT/cs766-project). This is the repository you are currently in.

---

### Website and Demo Repository

The website repository, inference demos, sample images and presentation slides are all available in the [website and demo repository](https://github.com/AnteT/ffd-app), and slides can be found in this repo in the `presentation-slides` directory:

```text
Project
   └── presentation-slides/
       └── Artificial Facial Detection.pptx
```

Questions or comments? Please don't hesitate to reach out!

---

**Thank you!**
_Ante Tonkovic-Capin & Udhbhav Gupta_