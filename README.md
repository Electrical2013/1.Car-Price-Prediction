# Car Price Prediction 🚗💰

This project predicts the **selling price of used cars** based on their key attributes such as company, model, year, kilometers driven, and fuel type. The dataset is scraped from Quikr.com and processed through a full **machine learning lifecycle**: from **data cleaning and feature engineering** to **model training, evaluation, and deployment** with **FastAPI**.

## 📌 Project Overview

* **Objective:** Predict resale prices of cars to ensure fair valuation for buyers and sellers.
* **Problem Type:** Supervised Regression Task.
* **Algorithms Used:**

  * Linear Regression ✅ (Best Model)
  * Random Forest Regressor
  * XGBoost Regressor
* **Evaluation Metrics:** R² Score, MAE, RMSE.
* **Deployment:** FastAPI REST API for real-time inference.

---

## 📂 Repository Structure

```
├── app.py                     # FastAPI app for model deployment
├── Car Price Prediction.ipynb # Jupyter Notebook with full ML lifecycle
├── Cleaned_Car_data.csv       # Preprocessed dataset
├── LinearRegressionModel.pkl  # Saved Linear Regression model pipeline
├── Project Details.word       # Detailed project report
├── quikr_car.csv              # Raw dataset
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone the repository:

   ```bash
git clone https://github.com/Electrical2013/1.Car-Price-Prediction.git

Change directory to the cloned repository
cd 1.Car-Price-Prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

    **requirements.txt:**

   ```
fastapi==0.116.1
numpy==2.3.3
pandas==2.3.2
pydantic==2.11.9
   ```

3. Run the FastAPI app:

   ```bash
   python -m uvicorn app:app --reload
   ```

4. Access API docs in browser:

   * Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🚀 Usage

### API Example

POST request to `/predict` with JSON body:

```json
{
  "name": "Maruti Suzuki Swift",
  "company": "Maruti",
  "year": 2019,
  "kms_driven": 100,
  "fuel_type": "Petrol"
}
```

Response:

```json
{
  "predicted_price": 455275.79
}
```

---

## 📊 Model Performance

| Model                   | R² Score (Test) |
| ----------------------- | --------------- |
| **Linear Regression**   | **0.87**        |
| XGBoost Regressor       | 0.86            |
| Random Forest Regressor | 0.75            |

* Linear Regression was selected as the final model due to its **accuracy and interpretability**.

---

## 🔍 Key Learnings

* Rigorous **data cleaning** (removing outliers, fixing inconsistent entries) is critical.
* **EDA insights** guided feature engineering (e.g., car\_age, log\_kms).
* **Linear Regression** surprisingly outperformed more complex models.
* **Deployment with FastAPI** enables real-time, production-ready predictions.

---

## 📌 Future Improvements

* Add more features (engine size, transmission, number of owners).
* Improve model retraining pipeline with continuous data updates.
* Deploy on cloud (AWS/GCP/Azure) with Docker for scalability.
* Build a front-end UI for non-technical users.

---

## ✍️ Author

**Eshita Adhikary**

If you find this project useful, ⭐ the repo and feel free to contribute!
