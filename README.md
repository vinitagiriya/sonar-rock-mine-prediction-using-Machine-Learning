# 🔍 Sonar Rock vs Mine Prediction using Machine Learning

A simple machine learning project using **Logistic Regression** to predict whether a sonar signal is bouncing off a **rock (R)** or a **mine (M)**. Built in **Google Colab** with **NumPy**, **Pandas**, and **Scikit-learn**, this project includes data preprocessing, model training, evaluation, and prediction.  
Dataset used: [UCI Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))

---

## 📁 Dataset

- **Name**: Sonar dataset  
- **Format**: CSV  
- **Features**: 60 numeric values (frequency band energy)  
- **Target**:  
  - `R` = Rock  
  - `M` = Mine

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Google Colab

---

## 📌 Project Workflow

This project follows these five main steps:

1. **Data Collection & Exploration**  
   - Loaded the sonar dataset using Pandas  
   - Displayed shape, head, and statistics using `.shape`, `.head()`, and `.describe()`  
   - Checked class distribution using `.value_counts()`  
   - Grouped data by class label to compare averages  

2. **Data Preprocessing**  
   - Separated features (`X`) and labels (`y`)  
   - Split data into training and testing sets using `train_test_split()`  
   - Used `stratify=y` to ensure balanced label distribution  

3. **Model Training – Logistic Regression**  
   - Created a `LogisticRegression` model  
   - Trained the model using:  
     ```python
     model = LogisticRegression()
     model.fit(x_train, y_train)
     ```

4. **Model Evaluation**  
   - Predicted on training and test data  
   - Calculated accuracy using `accuracy_score`  
     ```python
     x_train_prediction = model.predict(x_train)
     training_accuracy = accuracy_score(x_train_prediction, y_train)

     x_test_prediction = model.predict(x_test)
     test_accuracy = accuracy_score(x_test_prediction, y_test)
     ```

5. **Prediction System**  
   - Took a 60-feature input from a sonar signal  
   - Converted it to a NumPy array and reshaped it  
   - Made a prediction and printed whether it was a Rock or a Mine  
     ```python
     input_data = (0.0286, 0.0453, ..., 0.0062)
     input_array = np.array(input_data).reshape(1, -1)
     prediction = model.predict(input_array)

     if prediction[0] == 'R':
         print("The object is Rock")
     else:
         print("The object is Mine")
     ```

---

## 📈 Results

| Dataset      | Accuracy |
|--------------|----------|
| Training Set | ~86%     |
| Test Set     | ~81%     |

---

## 🚀 How to Use

1. Upload the dataset (`Copy of sonar data.csv`) to your Colab environment.  
2. Run each notebook cell in order.  
3. Modify the `input_data` array in the prediction section to test new values.

---

## 📦 Project Structure

├── sonar-rock-mine-prediction.ipynb
├── Copy of sonar data.csv
├── .gitignore
└── README.md

## 🙋‍♀️ Author

**Vinita Giriya**  
Made with ❤️ while learning Machine Learning and working on sonar signal classification.




