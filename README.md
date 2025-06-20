# 🧠 Parkinson's Disease Prediction Web Application

A machine learning-powered web application to predict Parkinson’s Disease and estimate its severity using patient vocal features. This project was developed as part of an academic ML project with additional improvements for better usability and performance.

---

## 📌 About the Project

This project uses **three machine learning models**:

1️⃣ **SVM Classifier** → Predicts if the patient has Parkinson’s (Yes / No) using vocal features.  

2️⃣ **Random Forest Regressor** → Predicts the **Total UPDRS** score (Unified Parkinson's Disease Rating Scale) for severity estimation.

3️⃣ **Random Forest Classifier** → Predicts **Severity Levels** (Mild / Moderate / Severe) based on the predicted UPDRS score.

Severity levels are calculated using the following range:

- 🟢 **Mild**: 0 - 20
- 🟡 **Moderate**: 21 - 40
- 🔴 **Severe**: 41 - 60

---

## 🌐 Features

- 🎯 Predict Parkinson’s Disease (Yes / No)
- 📊 Predict UPDRS score (severity score)
- 🚦 Display of severity level (Mild, Moderate, Severe)
- 🖥️ Clean & modern Bootstrap-based UI with interactive elements
- 🌀 Loading spinner for prediction feedback

---

## 📋 Technologies Used

- Python (Flask)
- HTML, CSS (Bootstrap 5)
- Machine Learning (Scikit-learn)

---

## 🤖 Models Training Notebook

You can find the model training code and approach in the provided Colab notebook inside this repository.

---

## 📂 Datasets Used

- [Parkinson’s Disease Dataset (for Diagnosis)](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)
- [Parkinson’s Telemonitoring Dataset (for Severity & UPDRS Score)](https://archive.ics.uci.edu/dataset/174/parkinsons)

---

## 📑 Citation

M. Little. "Parkinsons," UCI Machine Learning Repository, 2007. [Online]. Available: https://doi.org/10.24432/C59C74.

---

## 📎 License

This project is for academic and educational purposes.

---

