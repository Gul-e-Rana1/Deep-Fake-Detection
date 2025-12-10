# Deep Fake Detection using Vision Transformers (ViT)
This repository contains a Streamlit app that detects deep fake images using a fine-tuned Vision Transformer (ViT) model trained on the CIFAR-10 dataset.

## Overview
This project demonstrates the use of ViT for detecting deep fake images. The model is fine-tuned on CIFAR-10 dataset and integrated into a Streamlit app for user-friendly interaction.

## Features
- **Deep Fake Detection**: Uses the `nateraw/vit-base-patch16-224-cifar10` pre-trained model to detect deep fake images.
- **Face Detection**: Employs OpenCV's Haar Cascade classifier to detect faces in the uploaded images.
- **User Interface**: Interactive web interface built using Streamlit for uploading and analyzing images.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/Gul-e-Rana1/Deep-Fake-Detection.git
    cd Deep-Fake-Detection
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure you have the saved model file `vit_finetuned_cifar10.pth` in the project directory.

## Usage
1. Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open your browser and navigate to the provided local URL.
3. Upload an image and let the app detect whether it's a deep fake or not.

## How It Works
1. **Preprocessing**: The uploaded image is preprocessed to fit the input size of the model.
2. **Face Detection**: The app detects faces in the image using OpenCV.
3. **Deep Fake Detection**: The pre-trained ViT model predicts if the image is real or fake based on the processed input.

## Project Structure
- `app.py`: Main Streamlit app code.
- `requirements.txt`: List of required Python packages.

## Contributing
Feel free to fork this repository, submit pull requests, and file issues if you encounter any problems.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Hugging Face for the ViT model
- Streamlit for the interactive UI framework
- OpenCV for face detection
