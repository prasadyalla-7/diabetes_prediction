# Diabetes Prediction Using Machine Learning

## Overview

This project focuses on predicting whether a person is likely to have diabetes using machine learning classification algorithms. The project includes data preprocessing, exploratory data analysis, feature scaling, model training, and performance evaluation.

Multiple classification algorithms are implemented and compared to identify an effective model for diabetes prediction.

## Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains several medical attributes that are used to predict the diabetes outcome.

### Features

| Feature                  | Description                                     |
| ------------------------ | ----------------------------------------------- |
| Pregnancies              | Number of pregnancies                           |
| Glucose                  | Plasma glucose concentration                    |
| BloodPressure            | Diastolic blood pressure                        |
| SkinThickness            | Triceps skinfold thickness                      |
| Insulin                  | Serum insulin level                             |
| BMI                      | Body Mass Index                                 |
| DiabetesPedigreeFunction | Diabetes pedigree function                      |
| Age                      | Age of the patient                              |
| Outcome                  | Diabetes status: 0 = Non-diabetic, 1 = Diabetic |

## Project Workflow

The project follows these main steps:

1. Load and explore the dataset
2. Remove duplicate records
3. Handle invalid zero values
4. Perform exploratory data analysis
5. Analyze correlations between features
6. Separate input features and target variable
7. Apply feature scaling
8. Split the data into training and testing sets
9. Train multiple classification models
10. Evaluate and compare model performance
11. Predict the outcome for a new instance

## Data Preprocessing

### Removing Duplicate Records

Duplicate records are removed from the dataset to improve data quality.

### Handling Invalid Zero Values

Zero values in the following medical features are treated as missing or invalid values:

* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI

These zero values are replaced with the mean value of the corresponding feature.

### Feature Scaling

`StandardScaler` is used to standardize the input features before model training.

Standardization helps bring features with different numerical ranges to a common scale, which is particularly important for distance-based algorithms such as K-Nearest Neighbors.

## Exploratory Data Analysis

The project includes several visualization techniques to understand the dataset:

* Outcome distribution using pie chart and count plot
* Histograms of individual features
* Scatter matrix
* Pair plot
* Correlation heatmap

These visualizations help identify feature distributions, relationships between variables, and correlations with the target variable.

## Machine Learning Models

The following classification algorithms are implemented:

### K-Nearest Neighbors

A default KNN classifier is first trained and evaluated.

The model is then tested again using:

```python
KNeighborsClassifier(n_neighbors=25)
```

This allows the performance of different neighborhood settings to be compared.

### Logistic Regression

Logistic Regression is used as a classification model with the `liblinear` solver.

The model is implemented using the One-vs-Rest classification strategy.

### Gaussian Naive Bayes

Gaussian Naive Bayes is also trained and evaluated for diabetes prediction.

## Model Evaluation

The models are evaluated using:

* Training Accuracy
* Testing Accuracy
* Accuracy Score
* Confusion Matrix

Confusion matrices are generated for the classification models to visualize correct and incorrect predictions.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## How to Run

1. Place `diabetes.csv` in the same directory as the notebook.
2. Open the Jupyter Notebook:

```bash
jupyter notebook
```

3. Open `diabetes_prediction.ipynb`.
4. Run all cells in sequence.

## Project Structure

```text
diabetes-prediction/
│
├── diabetes_prediction.ipynb
├── diabetes.csv
└── README.md
```

## Prediction

The trained model can also be used to predict the diabetes outcome for a new patient instance.

The new input data is preprocessed and passed to the trained classifier, which predicts one of the following classes:

* `0` — Non-diabetic
* `1` — Diabetic

## Conclusion

This project demonstrates an end-to-end machine learning workflow for diabetes prediction. It covers data cleaning, exploratory data analysis, feature scaling, classification using multiple machine learning algorithms, and model evaluation.

The project provides a practical comparison of K-Nearest Neighbors, Logistic Regression, and Gaussian Naive Bayes for a binary classification problem.


