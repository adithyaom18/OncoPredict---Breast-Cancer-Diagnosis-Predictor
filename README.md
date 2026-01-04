          🩺 OncoPredict – Breast Cancer Diagnosis Predictor

## 📸 Screenshots

### Index Page
<img src="screenshots\index.png" alt="Index Page" width="400">

### Mean Features(INPUT)
<img src="screenshots\Mean Features.png" alt="" width="400">

### Standard Error Features(INPUT)
<img src="screenshots\Standard Error Features.png" alt="" width="400">

### Worst case Features(INPUT)
<img src="screenshots\Worst Case Features.png" alt="" width="400">

### Benign(OUTPUT)
<img src="screenshots\Prediction.png" alt="Sales By Category" width="400">

### Malignant(OUTPUT)
<img src="screenshots\Maligpre.png" alt="" width="400">

### Confusion Matrix ON Test Data
<img src="screenshots\Confusion Features.png" alt="" width="400">

### Model Performance
<img src="screenshots\Model Performance.png" alt="" width="400">

🧠 Overview: 

The Breast Cancer Diagnosis Predictor is a machine learning–powered web application built using Streamlit.
It predicts whether a breast tumor is Benign (0) or Malignant (1) based on 30 diagnostic features extracted from digitized images of Fine Needle Aspirates (FNA) of breast masses.

This project follows an industry-standard ML workflow:

1.Experimentation & hyperparameter tuning in Jupyter Notebook

2.Model comparison using consistent evaluation metrics

3.Best model selection based on test accuracy

4.Modular deployment of the selected model in a Streamlit application

🎯 Project Objective

To evaluate multiple machine learning algorithms using hyperparameter tuning, identify the best-performing model based on accuracy, and deploy the selected model using modular and production-ready code.

🚀 Key Features:

✅ Interactive slider + numeric inputs for all features

📊 Features grouped into:

1.Mean features

2.Standard Error features

3.Worst-case features

🧪 Hyperparameter tuning (GridSearchCV) for multiple algorithms

🏆 Automatic best model selection based on test accuracy

📈 Visualized performance metrics:

‣ Accuracy

‣ Precision

‣ Recall

‣ F1 Score

📊 Model performance comparison bar chart

🧊 Confusion matrix heatmap

🔍 Live predictions with confidence score

🧩 Clean modular coding structure

🤖 Machine Learning Models Evaluated

The following algorithms were trained and tuned during experimentation:

‣ Logistic Regression

‣ Random Forest

‣ Support Vector Machine (SVM)

‣ AdaBoost

‣ XGBoost

‣ CatBoost

After hyperparameter tuning and evaluation, the model with the highest test accuracy was selected and deployed.

🧠 Model Selection Strategy

All models were trained using the same preprocessing pipeline and evaluated using identical metrics to ensure a fair comparison.

• Hyperparameter tuning was performed using GridSearchCV

• Evaluation was done on a held-out test set

• The model with the highest test accuracy was selected as the final model

• Only the best model was used in the deployed application

This separation of experimentation (Jupyter Notebook) and deployment (modular code) follows real-world ML best practices.

🧬 Features Used for Prediction

A total of 30 tumor features are used:

📊 Mean Features

• radius_mean

• texture_mean

• perimeter_mean

• area_mean

• smoothness_mean

• compactness_mean

• concavity_mean

• concave points_mean

• symmetry_mean

• fractal_dimension_mean

📉 Standard Error Features

• radius_se

• texture_se

• perimeter_se

• area_se

• smoothness_se

• compactness_se

• concavity_se

• concave points_se

• symmetry_se

• fractal_dimension_se

⚠️ Worst-Case Features

• radius_worst

• texture_worst

• perimeter_worst

• area_worst

• smoothness_worst

• compactness_worst

• concavity_worst

• concave points_worst

• symmetry_worst

• fractal_dimension_worst

📂 Dataset

This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset, publicly available on Kaggle:

🔗 https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

⚠️ Disclaimer:
This dataset is used strictly for educational and research purposes.
The application is not intended for clinical or medical use.

📦 Installation

1️⃣ Create a virtual environment (recommended)
```bash
conda create -n breast-cancer-diagnosis python=3.10
```
Activate virtual environment
```bash
conda activate breast-cancer-diagnosis
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

💻 Usage
Run the Streamlit app
```bash
streamlit run app/main.py
```

• The app will open in your default browser

• Enter tumor features using sliders or numeric inputs

• View prediction results and confidence

• Explore model performance metrics and visualizations

🧪 Educational Disclaimer

⚠️ This project is developed for learning and demonstration purposes only.
It should not be used for medical diagnosis or clinical decision-making.

📫 Contact

📧 Email: kadithyaom@gmail.com

🔗 GitHub: https://github.com/adithyaom18/cancer-prediction

🌐 Live Demo: https://oncopredict---breast-cancer-diagnosis-predictor.streamlit.app/

