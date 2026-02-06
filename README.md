# Fake News Detection using Machine Learning

This project implements a robust NLP pipeline to classify news articles as either **Fake** or **True**. It utilizes various machine learning algorithms and compares their performance after hyperparameter tuning.

## Dataset
The dataset consists of approximately 45,000 labeled articles:
- **Fake.csv**: 23,481 articles.
- **True.csv**: 21,417 articles.
*Note: Due to file size limits, ensure these are placed in the root directory before running the notebook.*



## Technical Pipeline

### 1. Data Cleaning & Preprocessing
The notebook utilizes a custom `wordopt` function to:
- Convert text to **lowercase**.
- Remove **URLs** and **HTML tags**.
- Strip **punctuation** and digits.
- Remove **newline characters**.

### 2. Feature Extraction
Text data is converted into numerical vectors using **TF-IDF Vectorization** (`TfidfVectorizer`), which helps the models weigh the importance of words across the entire dataset.

### 3. Model Implementation & Tuning
The following models were implemented:
- **Logistic Regression**
- **Random Forest Classifier**
- **Decision Tree Classifier**
- **Linear SVM Classifier**
- **Multinomial Naive Bayes**
- **KNN**

The models were tuned using these optimization techniques:
- **RandomizedSearchCV**
- **GridSearchCV**

## Performance Results
Based on the final evaluation in the notebook, the models achieved the following scores:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Tree(GridSearch)** | **99.44%** | 0.99.44 | 0.99.44 | 0.99.44 |
| **Linear SVM(Baseline)** | **99.31%** | 0.99.31 | 0.99.31 | 0.99.31 |
| **Logistic Regression(RandomSearch)** | **99.03%** | 0.9904 | 0.9903 | 0.9903 |
| **Random Forest(GridSearch)** | **97.64%** | 0.9765 | 0.9764 | 0.9763 |
| **Multinomial NB(RandomSearch)** | **94.36%** | 0.9438 | 0.9436 | 0.9435 |


## Project Structure
- `FakeNewsDetection.ipynb`: Main Jupyter Notebook containing code and visualizations.
- `requirements.txt`: List of necessary Python libraries.
- `LICENSE`: MIT License information.

## How to Run
1. **Clone the repo:** `git clone https://github.com/Subhang15/FakeNewsDetection.git`
2. **Install Libraries:** `pip install -r requirements.txt`
3. **Data Setup:** Place `Fake.csv` and `True.csv` in the project folder.
4. **Execute:** Run all cells in `FakeNewsDetection.ipynb`.
