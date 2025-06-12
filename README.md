# Autism_Prediction_Detection
# Description
This project focuses on building a machine learning model to predict the likelihood of Autism Spectrum Disorder (ASD) in individuals based on questionnaire responses and basic demographic features. The model leverages supervised learning techniques to analyze behavioral traits and health history, helping in the early identification of autism.

The dataset includes features such as age, gender, jaundice history, family history of autism, and responses to ten behavioral screening questions. Various classification algorithms like Logistic Regression, Random Forest, and Support Vector Machines (SVM) were implemented and evaluated using metrics such as accuracy, precision, recall, and F1-score.

Key Features:

Data preprocessing and cleaning

Exploratory data analysis (EDA) using visualization tools

Training and comparison of multiple ML algorithms

Confusion matrix and classification report for evaluation

User input interface for real-time prediction (optional Streamlit/Tkinter)

Tech Stack:

Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Tools: Jupyter Notebook / Google Colab

Outcome:
Achieved accurate predictions that could assist medical professionals in early screening processes. The model is designed for educational and research purposes, demonstrating the power of machine learning in healthcare applications.

This project aims to develop a machine learning model that can predict whether an individual is likely to have Autism Spectrum Disorder (ASD) based on their responses to a screening questionnaire and other relevant attributes.

---

## 📌 Project Overview

- **Goal**: To predict autism using features like age, gender, ATEC score, and behavioral attributes.
- **Type**: Binary Classification (Yes / No for autism prediction)
- **Tech Stack**: Python, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## 🗂️ Dataset

The dataset contains responses to 10 questions and additional demographic data. Some features include:

- A1 to A10 (screening responses)
- Age
- Gender
- Jaundice (Yes/No)
- Family history of ASD
- Who is completing the test
- Result (Autism = 1 / No Autism = 0)

Source: [UCI Autism Screening Dataset](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult)

---

## ⚙️ Installation

Clone this repository and install required packages:

```bash
git clone https://github.com/yourusername/autism-prediction.git
cd autism-prediction
pip install -r requirements.txt

 
