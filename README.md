# Image to Number AI Program

This project creates an AI program that converts images of handwritten digits to numbers using a convolutional neural network (CNN).

## File Structure

- `data/`: Contains the script for loading and preprocessing the dataset.
- `model/`: Contains the script for creating the CNN model.
- `train/`: Contains the script for training the model.
- `app/`: Contains the Flask app for serving the model.
- `saved_model/`: Directory where the trained model will be saved.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## Setup

1. Clone the repository.
2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

To train the model, run:

```sh
python train/train_model.py
