# Sign Language Recognition Project

This repository contains experiments with different machine learning approaches on a sign language dataset. The primary goal is to recognise hand gestures corresponding to alphabetic letters, even though the limited variety of data makes it challenging for the models to generalise well. The project includes experiments with both image-based data and CSV-formatted data, the latter of which is suitable for real-time inference using a webcam.

## Project Overview

The dataset is formatted similarly to the classic MNIST dataset. Each example is a 28x28 grayscale image representing a sign language gesture for one of the alphabet letters (with some omissions due to gesture ambiguities). Two formats are provided:

- **CSV Format:** Contains a header row and rows of pixel values (label, pixel1, pixel2, ..., pixel784). There are 27,455 training cases and 7,172 test cases.
- **Image Format:** The original dataset consists of images organised in directories.

The CSV version has been utilised to implement a real-time inference pipeline using a webcam.

## Dataset Details

The dataset is designed to mimic the MNIST format:
- **Dimensions:** Each image is 28x28 pixels.
- **Format:** Grayscale images with pixel values ranging from 0 to 255.
- **Labels:** Each case maps to a letter in the alphabet. The labels (0-25) correspond to the letters A-Z, with certain letters (such as J and Z) omitted due to the complexities of their gesture motions.

The CSV version is particularly useful as it allows the model to be deployed for real-time inference, for example using a webcam to capture live input.

## Models and Experiments

Several models have been experimented with:
- **Convolutional Neural Network (CNN):**  
  The `cnn_model.py` file contains the definition and training code for the CNN approach. Confusion matrices generated at various epochs (e.g. epochs 5, 15, 20, 25, 30) are provided to evaluate performance over time.

- **LSTM Model:**  
  The `ltsm.py` file contains code for an LSTM (or a related RNN-based) approach. This model attempts to capture sequential patterns within the dataset.

- **Vision Transformer (ViT):**  
  The `sign_language_vit_model.h5` file contains weights for a Vision Transformer model that has been applied to the image data.

- **CSV-based Model for Real-Time Inference:**  
  The `best_sign_model.h5` was trained on the CSV version of the dataset. This model is optimised for deployment in real-time scenarios, such as capturing input from a webcam.

## Running the Project

### Prerequisites
- Python 3.11+
- [PyTorch](https://pytorch.org/) for CNN-based models
- [TensorFlow/Keras](https://www.tensorflow.org/) for models saved in `.h5` format
- Other dependencies such as NumPy, Pandas, Matplotlib, and OpenCV for image processing and real-time inference

### Installation

You can install the required packages using pip. For example:

```bash
pip install torch torchvision tensorflow numpy pandas matplotlib opencv-python
```

### Running the Code

- **Training the CNN Model:**  
  Run the `cnn_model.py` script to train and evaluate the CNN on your dataset.
  
  ```bash
  python cnn_model.py
  ```

  **May need data to work** 

- **Real-Time Inference with CSV-based Model:**  
  Use your preferred method (e.g. a separate inference script) to capture webcam input and process it using `best_sign_model.h5`. Ensure your webcam feed is preprocessed to match the CSV dataset format (28x28 grayscale).

- **Evaluating Results:**  
 `training_history.png`

## Results and Evaluation

The project includes multiple confusion matrices generated at different training epochs (5, 15, 20, 25, and 30) which help to evaluate how the model’s performance evolves over time. Despite experimenting with various architectures, the limited variety in the dataset poses a challenge for generalisation. This insight motivates further work in areas such as data augmentation or acquiring more diverse datasets.

## Future Work

- **Data Augmentation:** Enhance the dataset artificially by applying rotations, flips, and other transformations to improve generalisation.
- **Model Optimisation:** Experiment with more complex architectures or ensemble methods.
- **Data Collection:** Consider expanding the dataset to include more variation in hand gestures and backgrounds.

## Conclusion

This project represents a learning journey in applying different machine learning techniques to a challenging dataset. While the limited data variety restricts model generalisation, the experiments provide valuable insights and a solid foundation for further research and development in sign language recognition.
