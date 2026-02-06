# Fake News Detection using Machine Learning

This project implements a machine learning pipeline to classify news articles as either **Fake** or **True** based on their text content.

## Dataset
The project uses a dataset consisting of two files:
- **Fake.csv**: 23,481 fake news articles.
- **True.csv**: 21,417 true news articles.

Each article contains a title, body text, subject, and publication date.

## Features
- **Data Preprocessing**: Text cleaning and removal of punctuation/special characters.
- **Feature Extraction**: Uses `TfidfVectorizer` to convert text into numerical data.
- **Models Used**:
  - Logistic Regression (Randomized Search)
  - Random Forest (Grid Search)
  - Multinomial Naive Bayes (Randomized Search)
- **Visualization**: Includes data distribution and model performance plots using `seaborn` and `matplotlib`.

## Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9903** | 0.9904 | 0.9903 | 0.9903 |
| **Random Forest** | **0.9764** | 0.9765 | 0.9764 | 0.9763 |
| **Multinomial NB** | **0.9436** | 0.9438 | 0.9436 | 0.9435 |

## How to Run
1. Clone this repository.
2. Ensure you have the datasets `Fake.csv` and `True.csv` in the same directory as the notebook.
3. Install dependencies: `pip install -r requirements.txt`.
4. Open and run `MLProto1 (5) (3) (3)(1).ipynb` in Jupyter Notebook or Google Colab.
