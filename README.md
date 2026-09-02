# Enterprise Customer Churn & Retention Intelligence 📈

An end-to-end Machine Learning solution that predicts customer churn risk, uncovers primary risk drivers using **SHAP (SHapley Additive exPlanations)**, and recommends tailored retention strategies.

## Key Features
- **Engineered Domain Features:** Calculates `AvgCostPerMonth`, `TotalServices`, and `HasProtection` flags to improve predictive accuracy.
- **LightGBM Classification Model:** Handles class imbalance natively using balanced weights on historical subscriber behavior.
- **Explainable AI (SHAP Integration):** Provides visual waterfall plots showing feature impacts on churn scores.
- **Actionable Business Logic:** Automatically triggers custom retention offers based on risk tiers.

## Tech Stack
- **Language:** Python
- **Machine Learning:** LightGBM, Scikit-learn, SHAP
- **Data Manipulation:** Pandas, NumPy
- **Dashboard UI:** Streamlit, Matplotlib

## Quick Start
1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/Customer_Churn_Predictor.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run model training:
   python train_model.py
4. Launch dashboard:
   streamlit run app.py   
