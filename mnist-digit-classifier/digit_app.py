import numpy as np
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# load the trained mode
model = joblib.load('final_xgb_model.pkl')

CANVAS_SIZE = 280

def preprocess_image(image_data):
    """Convert the drawn digit into a 28x28 flatten and scaled image."""
    
    # convert the drawn image data to grayscale
    img = Image.fromarray(image_data).convert('L')
    resized_image = img.resize((28, 28), Image.NEAREST)  # Resize to 28x28
    
    # convert to numpy and flatten it to (1, 784)
    img_array = np.array(resized_image, dtype=np.float32).reshape(1, -1)
    
    # visualization of preprocessing steps
    st.write("### Image Preprocessing Steps")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(resized_image, cmap="gray")
    ax[1].set_title("Pixelated Image (28x28)")
    ax[1].axis("off")

    st.pyplot(fig)

    return img_array

def predict_digit(img_array):
    "Predict digit and class probabilities"
    pred_digit = model.predict(img_array)
    probabilities = model.predict_proba(img_array)[0]

    return pred_digit, probabilities

def main():
    st.title("Handwritten Digit Classifier")
    st.write("🖍 Draw a digit below (0-9) and let the model predict it!")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=26,
        stroke_color='white',
        background_color="black",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw" if mode else "transform",
        key="canvas"
    )

    if st.button("Predict Now"):
        if canvas_result.image_data is not None:
            try:
                # preprocess image
                processed_img = preprocess_image(canvas_result.image_data)

                # predict digit and class probabilities
                predicted_digit, probabilities = predict_digit(processed_img)
                
                # display prediction
                st.success(f"🧠 Model Prediction: {predicted_digit}")

                # plot probability distribution
                fig, ax = plt.subplots()
                ax.bar(range(10), probabilities, color="royalblue")
                ax.set_xticks(range(10))
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability")
                ax.set_title("Class Probability Distribution")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
