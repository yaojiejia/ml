import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle
import os
import random
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler



# Add class name prefix to each path based on class name include in filename
def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d", x).start()] + '/' + x)
    return df


def class_id_to_label(id):
    label_map = {1: 'glass', 2: 'paper', 3: 'cardboard', 4: 'plastic', 5: 'metal', 6: 'trash'}
    return label_map[id]


IMAGES_DIR = '/ml/dataset/garbage classification/Garbage classification'

train_file = '/ml/dataset/one-indexed-files-notrash_train.txt'
val_file = '/ml/dataset/one-indexed-files-notrash_val.txt'
test_file = '/ml/dataset/one-indexed-files-notrash_test.txt'

df_train = pd.read_csv(train_file, sep=' ', header=None, names=['path', 'label'])
df_valid = pd.read_csv(val_file, sep=' ', header=None, names=['path', 'label'])
df_test = pd.read_csv(val_file, sep=' ', header=None, names=['path', 'label'])

df_train = add_class_name_prefix(df_train, 'path')
df_valid = add_class_name_prefix(df_valid, 'path')
df_test = add_class_name_prefix(df_test, 'path')

df_train['label'] = df_train['label'].apply(class_id_to_label)
df_valid['label'] = df_valid['label'].apply(class_id_to_label)
df_test['label'] = df_test['label'].apply(class_id_to_label)

gen = ImageDataGenerator(rescale=1./255)    # rescaling the images between 0 and 1

gen_train = gen.flow_from_dataframe(
    dataframe=df_train,
    directory=IMAGES_DIR,
    x_col='path',
    y_col='label',
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True
)

gen_valid = gen.flow_from_dataframe(
    dataframe=df_valid,
    directory=IMAGES_DIR,
    x_col='path',
    y_col='label',
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True
)

test_gen = gen.flow_from_dataframe(
    dataframe=df_test,
    directory=IMAGES_DIR,
    x_col='path',
    y_col='label',
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

#
# def build_model(num_classes):
#     # Loading pre-trained ResNet model
#     base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
#
#     x = base_model.output
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dense(1024, activation='relu')(x)
#     predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
#
#     model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
#
#     base_model.trainable = False
#
#     return model
#
#
# model = build_model(num_classes=6)
#
# model.compile(optimizer='Adam',
#               loss='categorical_crossentropy',
#               metrics=[tf.keras.metrics.categorical_accuracy])
#
# model.summary()
# # Model training
#
# history = model.fit(
#     gen_train,
#     validation_data=gen_valid,
#     epochs=30
# )
# model_pkl_file = "model.pkl"
# with open(model_pkl_file, 'wb') as file:
#     pickle.dump(model, file)
# filenames = test_gen.filenames
# nb_samples = len(filenames)
#
# model.evaluate_generator(test_gen, nb_samples)
# test_x, test_y = test_gen.__getitem__(1)
# preds = model.predict(test_x)
# # Comparing predcitons with original labels
#

# Function to load and preprocess the image
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


