# Avocado-price-prediction
 AI
# 🥑 Avocado Price Prediction – AI Hackathon Project

A machine learning project developed during the Tech-A-Thon (Nov 9–10, 2024) to predict avocado prices using historical data and real-time inputs. This project helps vendors, producers, and market analysts make informed pricing decisions with a user-friendly web application.

---

## 📌 Project Description

The system uses historical avocado sales data to train machine learning models that predict the average price of avocados. Key features include:

- Input parameters: Region, Date, Total Volume, Total Bags, and bag types (Small, Large, X-Large).
- Real-time predictions through a Flask-based web interface.
- Final model: **Random Forest Regressor**, selected for its high accuracy and low error.

---

## 🧠 Machine Learning Models Used

Three models were trained and evaluated:

| Model                  | R² Score | MAE  |
|-----------------------|----------|------|
| Linear Regression      | 0.72     | 0.15 |
| Decision Tree Regressor| 0.81     | 0.12 |
| **Random Forest Regressor** | **0.89** | **0.10** |

> Random Forest was tuned using `RandomizedSearchCV` for hyperparameter optimization.

---

## 🔧 Technologies Used

### 🖥️ Programming & Libraries
- **Python 3.x**
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- joblib
- Flask (for backend)

### 🌐 Web Development
- HTML5, CSS3
- Bootstrap 5 (for styling)

---

## 🛠️ Project Structure

avocado-price-prediction/
├── app.py # Flask application backend
├── model.py # Model training and tuning
├── avocado.csv # Original dataset
│ └── index.html # Frontend form (HTML)
├── README.md # Project documentation
└── LICENSE # Project license

---

## 🖼️ Screenshots


- 🧹 Preprocessed dataset  screenshots/final dataset
- 📈 Model performance graphs  screenshots/graph
- 🌐 Web app interface with price prediction output screenshots/web

---

## 🧪 How to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/susa0676/avocado-price-prediction.git
   cd avocado-price-prediction
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Train the model (if needed)

bash
Copy
Edit
python model.py
Run the web app

bash
Copy
Edit
python app.py
Open your browser and go to http://localhost:5000 to use the app.

👥 Team Members
Sai Srinivasan M (22BIT0036)

Thrisheiyan U K (22BIT0559)

Sathish K (22BIT0672)

Utkrisht Arya (22BIT0190)

Sudharsanan G (22BIT0676)

Under the guidance of: Dr. S. Hemalatha
Institution: School of Computer Science Engineering & Information Systems

🎯 Key Learning Outcomes
Hands-on experience with regression algorithms and hyperparameter tuning.

End-to-end ML pipeline: data preprocessing, training, evaluation, and deployment.

Real-time ML application with Flask integration.

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.