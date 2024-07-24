import argparse
import pandas as pd
from catboost import CatBoostClassifier

# Load the trained model
MODEL_PATH = "cat_model.cbm"
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

def predict_churn(user_input):
    try:
        # Prepare data for prediction
        user_data = pd.DataFrame([user_input])
        
        # Make prediction
        prediction = model.predict_proba(user_data)[:, 1][0]
        
        return {"Churn Probability": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict churn probability.')
    parser.add_argument('--customerID', required=True, help='Customer ID')
    parser.add_argument('--gender', required=True, help='Gender')
    parser.add_argument('--senior_citizen', type=int, required=True, help='Senior Citizen (0/1)')
    parser.add_argument('--partner', required=True, help='Partner')
    parser.add_argument('--dependents', required=True, help='Dependents')
    parser.add_argument('--tenure', type=int, required=True, help='Tenure (months)')
    parser.add_argument('--phone_service', required=True, help='Phone Service')
    parser.add_argument('--multiple_lines', required=True, help='Multiple Lines')
    parser.add_argument('--internet_service', required=True, help='Internet Service')
    parser.add_argument('--online_security', required=True, help='Online Security')
    parser.add_argument('--online_backup', required=True, help='Online Backup')
    parser.add_argument('--device_protection', required=True, help='Device Protection')
    parser.add_argument('--tech_support', required=True, help='Tech Support')
    parser.add_argument('--streaming_tv', required=True, help='Streaming TV')
    parser.add_argument('--streaming_movies', required=True, help='Streaming Movies')
    parser.add_argument('--contract', required=True, help='Contract')
    parser.add_argument('--paperless_billing', required=True, help='Paperless Billing')
    parser.add_argument('--payment_method', required=True, help='Payment Method')
    parser.add_argument('--monthly_charges', type=float, required=True, help='Monthly Charges')
    parser.add_argument('--total_charges', type=float, required=True, help='Total Charges')

    args = parser.parse_args()

    new_customer_data = {
        "customerID": args.customerID,
        "gender": args.gender,
        "SeniorCitizen": args.senior_citizen,
        "Partner": args.partner,
        "Dependents": args.dependents,
        "tenure": args.tenure,
        "PhoneService": args.phone_service,
        "MultipleLines": args.multiple_lines,
        "InternetService": args.internet_service,
        "OnlineSecurity": args.online_security,
        "OnlineBackup": args.online_backup,
        "DeviceProtection": args.device_protection,
        "TechSupport": args.tech_support,
        "StreamingTV": args.streaming_tv,
        "StreamingMovies": args.streaming_movies,
        "Contract": args.contract,
        "PaperlessBilling": args.paperless_billing,
        "PaymentMethod": args.payment_method,
        "MonthlyCharges": args.monthly_charges,
        "TotalCharges": args.total_charges
    }

    # Predict churn probability using the model
    churn_probability = model.predict_proba(pd.DataFrame([new_customer_data]))[:, 1]

    # Format churn probability
    formatted_churn_probability = "{:.2%}".format(churn_probability.item())

    print(f"Churn Probability: {formatted_churn_probability}")
