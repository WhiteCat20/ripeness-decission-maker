import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing import image
import cv2
import easygui
from rembg import remove
from PIL import Image

# no bg


def model_1(input_path):
    interpreter = tf.lite.Interpreter(model_path=f'model/model 1.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open an image file dialog
    # input_path = easygui.fileopenbox(title='Select Image File')

    # Load the image using PIL
    img = Image.open(input_path)
    img = remove(img)

    # Convert PIL image to numpy array
    img = np.array(img)

    # Check if the image is loaded properly
    if img is None:
        print("Error: Image not loaded properly. Check the path and file format.")
    else:
        # Convert the image to RGB if it's not already
        if len(img.shape) == 2 or img.shape[2] == 1:  # grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Resize the image
        img = cv2.resize(img, (224, 224))

        # Convert the image to a numpy array and expand dimensions to match input shape
        X = np.expand_dims(img, axis=0)

        # Normalize the image data to match the input range expected by the model (usually 0-1)
        X = X / 255.0

        # Prepare the input tensor
        interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))

        # Invoke the interpreter
        interpreter.invoke()

        # Get the output
        val = interpreter.get_tensor(output_details[0]['index'])
        prediction = ''
        # Interpret the output
        if val[0][0] >= 0.5:
            prediction = val[0][0]
            # print(f'unripe {val[0][0]}')
        else:
            prediction = val[0][0]
            # print(f'ripe {val[0][0]}')
        return np.round(prediction, 2)

# bg


def model_2(input_path):
    interpreter = tf.lite.Interpreter(model_path=f'model/model 2.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open an image file dialog
    # input_path = easygui.fileopenbox(title='Select Image File')

    # Load the image using PIL
    img = Image.open(input_path)
    # img = remove(img)

    # Convert PIL image to numpy array
    img = np.array(img)

    # Check if the image is loaded properly
    if img is None:
        print("Error: Image not loaded properly. Check the path and file format.")
    else:
        # Convert the image to RGB if it's not already
        if len(img.shape) == 2 or img.shape[2] == 1:  # grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Resize the image
        img = cv2.resize(img, (224, 224))

        # Convert the image to a numpy array and expand dimensions to match input shape
        X = np.expand_dims(img, axis=0)

        # Normalize the image data to match the input range expected by the model (usually 0-1)
        X = X / 255.0

        # Prepare the input tensor
        interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))

        # Invoke the interpreter
        interpreter.invoke()

        # Get the output
        val = interpreter.get_tensor(output_details[0]['index'])
        prediction = ''
        # Interpret the output
        if val[0][0] >= 0.5:
            prediction = val[0][0]
            # print(f'unripe {val[0][0]}')
        else:
            prediction = val[0][0]
            # print(f'ripe {val[0][0]}')
        return np.round(prediction, 2)

# no bg


def model_3(input_path):
    interpreter = tf.lite.Interpreter(model_path=f'model/model 3.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open an image file dialog
    # input_path = easygui.fileopenbox(title='Select Image File')

    # Load the image using PIL
    img = Image.open(input_path)
    img = remove(img)

    # Convert PIL image to numpy array
    img = np.array(img)

    # Check if the image is loaded properly
    if img is None:
        print("Error: Image not loaded properly. Check the path and file format.")
    else:
        # Convert the image to RGB if it's not already
        if len(img.shape) == 2 or img.shape[2] == 1:  # grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Resize the image
        img = cv2.resize(img, (224, 224))

        # Convert the image to a numpy array and expand dimensions to match input shape
        X = np.expand_dims(img, axis=0)

        # Normalize the image data to match the input range expected by the model (usually 0-1)
        X = X / 255.0

        # Prepare the input tensor
        interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))

        # Invoke the interpreter
        interpreter.invoke()

        # Get the output
        val = interpreter.get_tensor(output_details[0]['index'])
        prediction = ''
        # Interpret the output
        if val[0][0] >= 0.5:
            prediction = val[0][0]
            # print(f'unripe {val[0][0]}')
        else:
            prediction = val[0][0]
            # print(f'ripe {val[0][0]}')
        return np.round(prediction, 2)
