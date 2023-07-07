import cv2
import numpy as np
import os
import tensorflow as tf


# load the pre-trained TensorFlow model
def get_model(path: str) -> dict:
    model = tf.keras.models.load_model(path)
    return {'model': model}


# function to classify target as with or without mask
def predict_mask(
    path: str, frame: np.ndarray, labels: list, resize_window: tuple
):
    # extract the target region from the frame and resize it to 224x224
    x, y = 50, 50
    resized_image = cv2.resize(frame, resize_window)
    resized_image = resized_image / resized_image.max()

    # make predictions on the target image
    model = get_model(path)['model']
    pred = model.predict(
        resized_image.reshape((1, resize_window[0], resize_window[1], 3)), verbose=0
    )[0]

    # get the predicted class label
    label = labels[np.argmax(pred)]
    color = (0, 255, 0)  # green
    print(pred, label)

    # display the label near the detected target
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    return pred, label


def predict_mask_with_yolo(
    path: str, frame: np.ndarray, labels: list, resize_window: tuple
):
    # extract the target region from the frame and resize it to 224x224
    step_size = 150
    w = 224
    all_labels = []
    all_preds = []
    all_coordinates = []
    for x in np.arange(50, frame.shape[0] - 224 - 50, step_size):
        for y in np.arange(50, frame.shape[1] - 224 - 50, step_size):
            cropped_image = frame[x : x + w, y : y + w, :]
            resized_image = cv2.resize(cropped_image, resize_window)
            resized_image = resized_image / resized_image.max()

            # make predictions on the target image
            model = get_model(path)['model']
            pred = model.predict(
                resized_image.reshape((1, resize_window[0], resize_window[1], 3)),
                verbose=0,
            )[0]

            # get the predicted class label
            label = labels[np.argmax(pred)]
            color = (0, 255, 0)  # green
            print(pred, label)

            # collect
            all_labels.append(label)
            all_preds.append(pred)
            all_coordinates.append((x, y, w))

    threshold = 0.9
    best_idx = np.argmax(
        [all_preds[ii][np.argmax(all_preds[ii])] for ii in range(len(all_preds))]
    )
    best_label = all_labels[best_idx]
    x, y, w = all_coordinates[best_idx]

    # display the label near the detected target
    cv2.putText(frame, best_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 1)

    return pred, label


def video_to_frames(video: str, path_output_dir: str):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    nom = video.split('.')[0]  # this [0] is to grab the file name; DO NOT CHANGE THIS
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, f'{nom}_{count}.png'), image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


def create_video(input_path, output_path):

    """
    Reads in png files from a folder and creates an mp4 video using OpenCV

    To use the code:
        create_video('/path/to/input/folder', '/path/to/output/video.mp4')
    """
    # Get all the png file names in the input path
    image_names = [img for img in os.listdir(input_path) if img.endswith('.png')]
    # Sort the file names based on their number prefix (assuming they have a consistent naming pattern)
    image_names_sorted = sorted(image_names, key=lambda x: x.split('.')[0])

    # Read in the first image to get the image size
    first_image = cv2.imread(os.path.join(input_path, image_names_sorted[0]))
    height, width, channels = first_image.shape

    # Set up the video writer with 30fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    # Loop through each image and write it to the video
    for image_name in image_names_sorted:
        image_path = os.path.join(input_path, image_name)
        img = cv2.imread(image_path)
        out.write(img)

    # Release the video writer and close all windows
    out.release()
    cv2.destroyAllWindows()


def take_picture(id: int):
    # Create a video capture object using the default camera
    cap = cv2.VideoCapture(1)

    # Check if the camera is opened correctly
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Capture the frame and display it
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press q to exit or s to save the current frame as an image file
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'image{id}.jpg', frame)
            print("Image saved!")
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


def draw_bounding_boxes(mask):
    # Convert mask to a 2D array
    mask = mask[:, :, 0]

    # Convert the mask data type to np.uint8
    mask = mask.astype(np.uint8)

    # Convert the mask to a 3-channel image
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Define the center 60% of the screen
    w, h = mask.shape
    x1, y1 = int(w * 0.2), int(h * 0.2)
    x2, y2 = int(w * 0.8), int(h * 0.8)

    # Draw a bounding box around the center 60% of the screen
    boxed_mask = cv2.rectangle(mask_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Find the center region mask
    center_region_mask = mask[y1:y2, x1:x2]

    # Find contours in the center region mask
    contours, _ = cv2.findContours(center_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around different numbers in the middle
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x += x1
        y += y1
        cv2.rectangle(boxed_mask, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return boxed_mask