from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# configuring augmentation function with angle to be rotated. More augmentation functions can be implemented
def augmentation_config(rotation_range):
    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    return aug


# generation augmentation images
def generateimages(image, aug, threshold, directory):
    total = 0
    os.mkdir(directory)
    img = tf.keras.preprocessing.image.load_img(image)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # generate only 9 images

    imageGen = aug.flow(
        img,
        batch_size=1,
        save_prefix="image",
        save_to_dir=directory,
        save_format="jpg",
    )

    for image in imageGen:
        total += 1

        if total == threshold:
            break


if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title("Image Augmentation Library")
    upload_file = st.file_uploader("Choose an image", type="jpg")
    #create temp_file to store uploaded image
    temp_file = NamedTemporaryFile(delete=False)
    if upload_file:
        temp_file.write(upload_file.read())
        image = Image.open(upload_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
        st.write("")
        rotating_angle = st.slider(
            "Select an augmentation angle to rotate", 0.0, 360.0, step=15.0
        )
        st.write("Rotation angle is", rotating_angle)

        threshold = st.slider(
            "Select no of images to generate", 1, 5
        )

        d = temp_file.name
        # print(d)
        file = d.split("/tmp/")[1]
        
        augmentations = augmentation_config(rotation_range=rotating_angle)
        generateimages(temp_file.name, augmentations, threshold=threshold,directory=file)

        st.beta_container()
        
        # bug- can't display the created images in the folder
        for p in Path(file).iterdir():
            print(os.path.join(file,p.name))
            orig = Image.open(os.path.join(file,p.name))
            st.image(orig, use_column_width=True)
