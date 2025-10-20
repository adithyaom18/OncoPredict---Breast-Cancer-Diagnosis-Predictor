          🩺 Breast Cancer Diagnosis Predictor

🧠 Overview: 

The Breast Cancer Diagnosis Predictor is a machine learning-powered web application built with Streamlit. It helps users predict whether a breast tumor is benign or malignant based on 30 diagnostic features derived from digitized images of fine needle aspirates (FNA) of breast masses.

The app offers:

• Interactive inputs for mean, standard error, and worst-case tumor features

• Multiple ML model choices: Logistic Regression, Random Forest, and SVM

• Visualized model performance metrics (accuracy, precision, recall, F1 score)

• A model comparison bar chart

• Display of best hyperparameters for each model (from GridSearchCV)

• Live predictions with model confidence

• Confusion matrix heatmap of the selected model

⚠️ This project was developed for educational purposes only using the Breast Cancer Wisconsin (Diagnostic) dataset. It is not intended for clinical use.

📂 Dataset
We used the Breast Cancer Wisconsin (Diagnostic) Data Set, which is publicly available on Kaggle:

📌 https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

🚀 Live Demo
A live version of the app can be found on Streamlit Community Cloud.

📦 Installation
To set up the project locally:

1. Create virtual environment (recommended):

conda create -n breast-cancer-diagnosis python=3.10
conda activate breast-cancer-diagnosis

2. Install dependencies:

pip install -r requirements.txt
This will install packages like streamlit, numpy, pandas, scikit-learn, matplotlib, seaborn, and plotly.

💻 Usage
To launch the app:

streamlit run app/main.py


• The app will open in your default web browser

• Enter tumor features using sliders or number inputs

• Choose any ML model to see the prediction and confidence

• Compare model performance visually

📂 Project Structure:

streamlit-cancer-predict-main/
│
├── app/
│   └── main.py              # Main Streamlit app
│
├── model/
│   ├── logreg.pkl           # Trained Logistic Regression model
│   ├── rf.pkl               # Trained Random Forest model
│   ├── svm.pkl              # Trained SVM model
│   ├── scaler.pkl           # StandardScaler
│   ├── imputer.pkl          # SimpleImputer
│   ├── metrics.pkl          # Model evaluation metrics
│   ├── scores.pkl           # Cross-validation scores
│   └── best_params.pkl      # Best hyperparameters from tuning
│
├── assets/
│   └── style.css            # Custom CSS for styling
│
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation


🧠 Features Used in Prediction:

• Mean: radius_mean, texture_mean, area_mean, ...

• Standard Error: radius_se, texture_se, ...

• Worst-case: radius_worst, area_worst, ...

A total of 30 tumor features are used to train and make predictions.


📫 Contact
📧 Email: kadithyaom@gmail.com
🔗 GitHub: https://github.com/adithyaom18/cancer-prediction.git