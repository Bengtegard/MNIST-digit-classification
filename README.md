### **Handwritten Digit Recognition using Machine Learning (XGBoost)**

#### **Overview**

This project implements a handwritten digit recognition system using **XGBoost** trained on the **MNIST dataset**. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each **28x28 pixels** in size.\
Instead of using **Deep Learning (CNNs)**, this project leverages **XGBoost**, a powerful gradient boosting algorithm, to achieve **98.58% accuracy** on the test set.

### **Demo**

![image](https://github.com/user-attachments/assets/8c3b4360-54c3-4ea2-b7a3-0b265e568705)



#### **Dataset**

- **MNIST Dataset**: A collection of handwritten digits from **0-9**.
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28x28 pixels (grayscale)

#### **Requirements**

Ensure you have Python 3.5+ and install the following dependencies:

```bash
pip install -r requirements.txt
```

Or use **Conda**:

```bash
conda env create -f environment.yml
conda activate ml_digit_streamlit
```

#### **Dependencies (requirements.txt)**

```
matplotlib==3.10.0
numpy==2.2.2
pandas==2.2.3
pillow==11.1.0
python==3.12.9
scikit-learn==1.6.1
scipy==1.15.1
streamlit==1.42.0
streamlit-drawable-canvas==0.9.3
xgboost==2.1.4
```

#### **Project Structure**

```
mnist-digit-classifier
│── mnist_digit/           # Project folder
│   │── digit_app.py        # Streamlit app for user interaction
│   │── model.py            # Script to train & save XGBoost model
│   │── train_eda_evaulate.ipynb  # Jupyter Notebook for experimentation
│   │── final_xgb_model.pkl  # Saved trained model
│── requirements.txt        # List of dependencies
│── README.md               # Project documentation (this file)
```

#### **Model Description**

This project uses **XGBoost (Extreme Gradient Boosting)** instead of CNNs.

- **XGBoost** is an efficient and scalable gradient boosting library.
- It achieved **98.58% accuracy** on the MNIST dataset.
- The model is trained using a **pipeline** with `StandardScaler` for feature scaling.
- **Augmentation** is applied using **shift transformations** to enhance generalization.

#### **How to Run**

1. **Train the Model**\
   If you want to retrain the model, run:

   ```bash
   python model.py
   ```

   This will save the trained model as **`final_xgb_model.pkl`**.

2. **Run the Streamlit App**\
   To launch the digit recognition app, run:

   ```bash
   streamlit run digit_app.py
   ```

3. **Test the Model**\
   You can load and test the trained model in Python:

   ```python
   import joblib
   import numpy as np

   model = joblib.load("final_xgb_model.pkl")
   sample_digit = np.random.rand(1, 784)  # Example test digit
   prediction = model.predict(sample_digit)
   print("Predicted digit:", prediction[0])
   ```

#### **Results**

- **Training Accuracy**: **96.13%**
- **Test Accuracy**: **98.58%** (higher than some CNN models)
- **Prediction Time**: **Fast inference using XGBoost**
- **Augmentation**: Improved robustness with shift transformations

#### **Why XGBoost instead of CNN?**

- **Faster Training**: XGBoost trains much faster than CNNs on smaller datasets.
- **No Need for a GPU**: CNNs require a GPU for efficient training; XGBoost works well on CPUs.
- **Competitive Accuracy**: Achieved **98.58% test accuracy**, comparable to CNN models.

#### **Future Improvements**

- Test performance with **Hyperparameter Optimization**.
- Extend the app to **accept real-time handwriting inputs**.
- Experiment with **different boosting techniques** (e.g., LightGBM, CatBoost).
