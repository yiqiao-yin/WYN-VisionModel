import cv2
import numpy as np
import os
import imageio as iio
import matplotlib.pyplot as plt
from PIL import Image


def convert_heic_to_image(file_path, output_format="png"):
    """
    Converts a HEIC image file to either PNG or JPG format.

    Parameters:
        file_path (str): The path to the input HEIC file.
        output_format (str): The output image format, either "png" or "jpg".
                             Defaults to "png".

    Returns:
        str: The path to the output image file.
    """
    # open the HEIC file and convert it to RGB format
    with Image.open(file_path) as im:
        rgb_im = im.convert("RGB")

        # set the output file extension based on the desired format
        if output_format.lower() == "png":
            output_extension = ".png"
        elif output_format.lower() == "jpg":
            output_extension = ".jpg"
        else:
            raise ValueError(
                "Unsupported output format. Must be either 'png' or 'jpg'."
            )

        # construct the output file path and save the converted image
        output_path = file_path
        output_path = file_path.replace(".HEIC", output_extension)
        rgb_im.save(output_path)

        # return the path to the output file
        return output_path


def conv_str_to_int(l: list) -> list:
    output = [np.where(np.array(list(dict.fromkeys(l))) == e)[0][0] for e in l]
    return output


def plot_image_with_pixel_values(
    image_array: np.array, fig_w: int, fig_h: int, fontsize: int
):
    # Set figure size
    if fig_w and fig_h:
        plt.figure(figsize=(fig_w, fig_h))

    # Plot the image
    plt.imshow(image_array)

    # Get dimensions of the image
    height, width, _ = image_array.shape

    # Add text annotations for each pixel
    for y in range(height):
        for x in range(width):
            # Get the RGB color values for the current pixel
            r, g, b = image_array[y, x]

            # Set the font color depending on the brightness of the pixel
            brightness = int(np.mean([r, g, b]))
            if brightness > 120:
                font_color = 'black'
            else:
                font_color = 'white'

            # Create the text string
            text_string = f'{r:.0f}, \n{g:.0f}, \n{b:.0f}'

            # Add the text annotation to the plot
            plt.text(
                x,
                y,
                text_string,
                color=font_color,
                fontsize=fontsize,
                ha='center',
                va='center',
            )

    # Show the plot
    plt.show()


def vid_to_png(input_path, output_path):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file: " + input_path)

    # Loop through each frame of the video
    i = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If there are no more frames, exit the loop
        if not ret:
            break

        # Write the current frame to a PNG file in the output folder
        filename = os.path.join(output_path, f"frame_{i:04}.png")
        cv2.imwrite(filename, frame)

        # Increment the frame counter
        i += 1

    # Release the video file handle
    cap.release()


def concatenate_images(folder_path):
    X = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            image = cv2.imread(os.path.join(folder_path, file_name))
            image = cv2.resize(image, (224, 224))
            X.append(image)

    return np.array(X)


def zero_out_margin(image_array, w):
    # Create a new array of zeros with the same size as the input image array
    output_array = np.zeros_like(image_array)
    
    # Copy the central region of the input image array into the output array
    output_array[w:-w, w:-w] = image_array[w:-w, w:-w]
    
    return output_array


def flatten_list(nested_list):
    """
    Flatten a nested list into a single list.
    
    Args:
        nested_list (list): A nested list.
        
    Returns:
        list: The flattened list.
    """
    flat_list = []
    
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
            
    return flat_list


def conv_str_to_int(l: list) -> list:
    output = [np.where(np.array(list(dict.fromkeys(l))) == e)[0][0] for e in l]
    return output


def plot_image_with_pixel_values(
    image_array: np.array, fig_w: int, fig_h: int, fontsize: int
):
    # Set figure size
    if fig_w and fig_h:
        plt.figure(figsize=(fig_w, fig_h))

    # Plot the image
    plt.imshow(image_array)

    # Get dimensions of the image
    height, width, _ = image_array.shape

    # Add text annotations for each pixel
    for y in range(height):
        for x in range(width):
            # Get the RGB color values for the current pixel
            r, g, b = image_array[y, x]

            # Set the font color depending on the brightness of the pixel
            brightness = int(np.mean([r, g, b]))
            if brightness > 120:
                font_color = 'black'
            else:
                font_color = 'white'

            # Create the text string
            text_string = f'{r:.0f}, \n{g:.0f}, \n{b:.0f}'

            # Add the text annotation to the plot
            plt.text(
                x,
                y,
                text_string,
                color=font_color,
                fontsize=fontsize,
                ha='center',
                va='center',
            )

    # Show the plot
    plt.show()