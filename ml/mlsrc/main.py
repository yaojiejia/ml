import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import random
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import modelHandler
import pickle
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

with open("model.pkl", 'rb') as file:
    model = pickle.load(file)
def load_and_preprocess_image(file_path, target_size=(224, 224)):
    img = Image.open(file_path)
    img = img.resize(target_size)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalize pixel values
    return img_array

# Function to make predictions
def predict_image(model, file_path, gen_train):
    img = load_and_preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    labels = gen_train.class_indices
    labels = {v: k for k, v in labels.items()}
    return labels[predicted_class], prediction

# Function to display the prediction
def display_prediction(file_path, model, gen_train):
    predicted_class, prediction = predict_image(model, file_path, gen_train)
    img = Image.open(file_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title('Predicted: {} \nConfidence: {:.2f}%'.format(predicted_class, np.max(prediction)*100))
    plt.axis('off')
    plt.show()
    return predicted_class

class FileCreatedHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:  # Check if the event is not for a directory

            print(display_prediction(event.src_path, model, modelHandler.gen_train))
            time.sleep(6)

def monitor_directory(directory):
    event_handler = FileCreatedHandler()
    observer = Observer()
    observer.schedule(event_handler, path=directory, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    directory_to_watch = "/home/alex/Documents/ml/ml/test"  # Specify the directory you want to monitor
    monitor_directory(directory_to_watch)
