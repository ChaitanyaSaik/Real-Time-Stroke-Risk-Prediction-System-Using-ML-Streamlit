#  Real-Time Stroke Risk Prediction System Using ML & Streamlit

A web application built using **Streamlit** and **Machine Learning** to predict the likelihood of a stroke based on health-related input features. This app also provides detailed insights into the dataset, preprocessing steps, exploratory data analysis (EDA), model training, evaluation metrics, and prediction results.

---

## 🌐 Live Demo

🚀 **Try the App Now on Streamlit Cloud**  
🔗 [https://your-streamlit-app-url.streamlit.app](https://heart-stroke-prediction-app-nfzdxcyldqxy3cuppqfvac.streamlit.app/)  
> *(Replace with your actual app link above)*

---

## 🚀 Features

- ✅ **Dataset Overview** with interactive tables and charts  
- 🧹 **Data Preprocessing**: Handling missing values, encoding, scaling  
- 📊 **Data Analysis & Insights**: Visualizations using matplotlib and seaborn  
- 🤖 **ML Models Used**:  
  - Logistic Regression  
  - Random Forest Classifier  
  - XGBoost Classifier  
  - Support Vector Machine (SVM)  
  - K-Nearest Neighbors (KNN)  
- ⚖️ **SMOTE** for handling imbalanced data  
- 📈 **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- 🧠 **Stroke Prediction**: Real-time prediction based on user input  
- 💡 **Clean UI** powered by Streamlit  

---

## 🧬 Dataset

The dataset used in this app is from [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) and includes features like:

- Gender
- Age
- Hypertension
- Heart Disease
- Marital Status
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status
- Stroke (Target variable)

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/heart-stroke-prediction-app.git
cd heart-stroke-prediction-app
```
Create and activate a virtual environment (optional but recommended)
```bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install dependencies

```bash

pip install -r requirements.txt
```
🏃‍♂️ Running the App Locally
```bash

streamlit run app.py
Make sure app.py contains the main Streamlit application logic.
```
---

**Sample Screenshots**
📂 Dataset Preview

📉 Feature Distributions

📈 Correlation Heatmap

🧠 Model Performance Reports

🩺 Real-time Stroke Prediction UI

---

🔍 **Folder Structure**
```bash
├── app.py
├── requirements.txt
├── README.md
├── dataset/
│   └── stroke_data.csv
├── models/
│   └── trained_model.pkl
├── utils/
│   └── prediction_system.py
```

---
📊 **Model Evaluation**
Each model was evaluated using the following metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

Confusion Matrix

---

🛡️ **Note on Imbalanced Data**
This dataset is imbalanced (majority of patients did not have a stroke). Therefore, SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the training data and improve model fairness.

---

✨ **Future Improvements**
Add feature importance visualizations

Allow CSV upload for batch prediction

Incorporate deep learning models (DNN, GNN)

Enable model comparison plots

Deploy using Docker or HuggingFace Spaces

---
🙌 **Acknowledgements**
Streamlit

Scikit-learn

Imbalanced-learn

Kaggle Dataset

---
**Contact**
Author: Chaitanya Sai Kurapati
Email: [your-email@example.com]
LinkedIn: linkedin.com/in/yourprofile


