import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data.mnist_data import load_and_preprocess_data
from model.cnn_model import create_model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = create_model()
    
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    
    model.save('saved_model/mnist_model.keras')

if __name__ == '__main__':
    train_model()
