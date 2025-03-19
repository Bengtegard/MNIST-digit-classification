import numpy as np
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import seaborn as sns

# Load the trained model
model = joblib.load('final_xgb_model.pkl')

CANVAS_SIZE = 340

def preprocess_image(image_data):
    """Convert the drawn digit into a 28x28 flatten and scaled image."""
    img = Image.fromarray(image_data).convert('L')

    # invert colors (black to white, white to black)
    img_invert = ImageOps.invert(img)

    # crop the image to the bounding box of the digit
    bbox = img_invert.getbbox()

    if bbox:
        left, upper, right, lower = bbox

        # add padding around the digit
        padding = 25  

        left = max(left - padding, 0)
        upper = max(upper - padding, 0)
        right = min(right + padding, img_invert.width)
        lower = min(lower + padding, img_invert.height)

        # crop the image with the expanded bounding box
        img_cropped = img_invert.crop((left, upper, right, lower))
    else:
        img_cropped = img_invert  # ff bbox is None, use the full image
        img_cropped = img_invert.crop(bbox)

    # get the size of the cropped image
    cropped_width, cropped_height = img_cropped.size

    # resize the image so that the longer side fits into 28 while maintaining aspect ratio
    scale_factor = min(28 / cropped_width, 28 / cropped_height)
    new_width = int(cropped_width * scale_factor)
    new_height = int(cropped_height * scale_factor)

    img_resized = img_cropped.resize((new_width, new_height), Image.LANCZOS)

    # create a 28x28 black canvas
    canvas = Image.new('L', (28, 28), color=0)

    # calculate position to center the resized image on the 28x28 canvas
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2

    # paste the resized image into the center of the canvas
    canvas.paste(img_resized, (left, top))

    # convert the final image to numpy array and flatten
    img_array = np.array(canvas, dtype=np.float32).reshape(1, -1)
    
    # visualization of preprocessing steps
    st.write("### Image Preprocessing Steps")
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(img_invert, cmap='gray')
    ax[1].set_title("Inverted Image")
    ax[1].axis("off")
    ax[2].imshow(canvas, cmap="gray")
    ax[2].set_title("Final Image (28x28)")
    ax[2].axis("off")
    st.pyplot(fig)
    
    return img_array

def predict_digit(img_array):
    """Predict digit and class probabilities with confidence intervals."""
    pred_digit = model.predict(img_array)[0]
    probabilities = model.predict_proba(img_array)[0]
    confidence = np.max(probabilities)
    return pred_digit, probabilities, confidence

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Project Description", "Demo"])
    
    if page == "Demo":
        st.title("Digit Classifier ✍️")
        st.write("🖍 Draw a digit below (0-9) and let the model predict it!")

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=23,
            stroke_color='black',
            background_color="white",
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("Predict Now"):
            if canvas_result.image_data is not None:
                try:
                    processed_img = preprocess_image(canvas_result.image_data)
                    predicted_digit, probabilities, confidence = predict_digit(processed_img)
                    
                    st.success(f"🧠 Model Prediction: {predicted_digit} (Confidence: {confidence:.2%})")
                    
                    # plot probability distribution with confidence intervals
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=list(range(10)), y=probabilities, palette="coolwarm", ax=ax)
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Probability")
                    ax.set_title("Class Probability Distribution")
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    elif page == "Project Description":
        st.title("About this App")
        st.write("This project implements a handwritten digit recognition system using XGBoost trained on the MNIST dataset.")
        st.write("The MNIST dataset consists of 70,000 grayscale images of      handwritten digits (0-9), each 28x28 pixels in size.")
        st.write("Instead of using Deep Learning (CNNs), this project leverages XGBClassifier, a powerful gradient boosting algorithm, and it resulted in 98.58% accuracy on the test set.")
        st.write("Developed using Streamlit.")

if __name__ == "__main__":
    main()
