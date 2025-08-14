# ML Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)

A web-based application for predicting **Diabetes** and **Heart Disease** using **Machine Learning**. Built using **Python**, **Streamlit**, and **pre-trained ML models**.

## Features

- Predict Diabetes based on user health parameters.
- Predict Heart Disease based on user health parameters.
- Sample input values provided for quick testing.
- Easy-to-use web interface.
- Real-time prediction results.

## Technologies Used

- **Python**
- **Streamlit**
- **scikit-learn (for ML models)**
- **pickle (for saving/loading ML models)**
- **streamlit-option-menu (for sidebar navigation)**

## 🖥️ Live Demo
[🌐 Try it here](https://ml-disease-predict.streamlit.app/)

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo> 
```
2. Create a Virtual Environment:
```bash
python -m venv myenv
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate     # Windows
```

3. Install required packages:

```bash

pip install -r requirements.txt
```

4. Run the app:

```bash

streamlit run app.py
```
---

## Usage
1. Open the app in your browser.

2. Use the sidebar to select Diabetes or Heart Disease Prediction.

3. Enter your health parameters or use the pre-filled sample values.

4. Click the Test Result button to get the prediction.

---

## Project Structure

```
ML-Disease-Prediction/
├── app.py
├── Saved Models/
│   ├── diabetes_model.sav
│   └── heart_model.sav
├── requirements.txt
└── README.md
```

--- 

## Deployment
Deployed on **Streamlit Cloud**.

Any changes pushed to GitHub automatically update the live app.

---

## 👤 Author
**Jishan Khan**
💼 [LinkedIn](https://www.linkedin.com/in/jishan-khan-ba5880342) | 💻 [GitHub](https://github.com/jishandev134)