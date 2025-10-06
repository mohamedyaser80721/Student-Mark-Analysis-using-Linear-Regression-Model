# Student Performance Prediction

This project is a **machine learning model** that predicts the **performance index of students** based on various factors like study hours, previous scores, sleep hours, extracurricular activities, and sample question papers practiced. The model uses **Linear Regression** and can be saved and loaded using `joblib` for future predictions.

---

## 📊 Dataset

The dataset used is `Student_Performance.csv`, which contains the following columns:

- `Hours Studied` — Number of hours a student studies.
- `Previous Scores` — Student’s scores in previous exams.
- `Extracurricular Activities` — Participation in extracurricular activities (Yes/No).
- `Sleep Hours` — Average sleep hours per day.
- `Sample Question Papers Practiced` — Number of sample papers practiced.
- `Performance Index` — Target variable representing student performance.

---

## ⚙️ Features

1. Data Preprocessing:
   - Converts categorical columns (`Extracurricular Activities`) into numeric using `LabelEncoder`.
   
2. Feature Selection:
   - Input features (`X`): `Hours Studied`, `Previous Scores`, `Extracurricular Activities`, `Sleep Hours`, `Sample Question Papers Practiced`.
   - Target (`y`): `Performance Index`.

3. Model:
   - Linear Regression (`sklearn.linear_model.LinearRegression`).
   - Trained on 80% of the dataset and tested on 20%.

4. Saving and Loading Model:
   - Model is saved using `joblib.dump()`.
   - Model can be loaded using `joblib.load()` for new predictions.

---

## 💻 How to Use

1. Clone the repository:
```bash
git clone <repository-url>
