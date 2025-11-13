import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from  BusinessMethod import BusinessMethod

data = pd.read_csv('https://drive.google.com/drive/u/1/my-drive/emi_prediction_dataset (1).csv')


#data.insert(0, 'ID', range(1, len(data) + 1))
#top10 = data.head(10)
#print(top10)
data["education"] = data["education"].fillna("Unknown")
data["monthly_rent"] = data["monthly_rent"].fillna(data["monthly_rent"].mean())
data["credit_score"] = data["credit_score"].fillna(data["credit_score"].mean())
data["emergency_fund"] = data["emergency_fund"].fillna(data["emergency_fund"].mean())
data["bank_balance"] = data["bank_balance"].str.replace(".0", "").astype(float)
data["bank_balance"] = data["bank_balance"].fillna(data["bank_balance"].mean())
data["monthly_salary"] = data["monthly_salary"].astype(str).str.replace(".0", "", regex=False)
data["monthly_salary"] = pd.to_numeric(data["monthly_salary"], errors="coerce")
data["monthly_salary"].fillna(data["monthly_salary"].mean(), inplace=True)
# Convert to string (NaN becomes 'nan'), replace ".0.0", then convert numeric
data["age"] = data["age"].astype(str).str.replace(".0.0", "", regex=False).astype(float)
data["age"] = data["age"].astype(int)
data["monthly_rent"] = data["monthly_rent"].fillna(data["monthly_rent"].mean())
data = data[data["credit_score"].between(300, 850)]
#row_count = len(data)
#print("Total rows:", row_count)

data  = data[data["current_emi_amount"] < 0.6 * data["monthly_salary"]]
data["emi_to_income_ratio"] = round(data["max_monthly_emi"] / data["monthly_salary"],2)
data["DTI"] = round(data["current_emi_amount"]/data["monthly_salary"],2)
data["ETI"] = round((data["school_fees"]+data["college_fees"]+data["travel_expenses"]+data["groceries_utilities"]+data["other_monthly_expenses"])/data["monthly_salary"],2)
data["affordability ratios"] = round((data["current_emi_amount"]+data["school_fees"]+data["college_fees"]+data["travel_expenses"]+data["groceries_utilities"]+data["other_monthly_expenses"])/data["monthly_salary"],2)



data["gender"] = data["gender"].replace("female","Female")
data["gender"] = data["gender"].replace("male","Male")
data["gender"] = data["gender"].replace("M","Male")
data["gender"] = data["gender"].replace("MALE","Male")
data["gender"] = data["gender"].replace("F","Female")
data["gender"] = data["gender"].replace("FEMALE","Female")
#distinct_genders = data["gender"].unique()
#print(distinct_genders)

bm = BusinessMethod()
data["credit_risk_score"] = data.apply(
    lambda row: bm.calculate_credit_risk_score(row["credit_score"], row["current_emi_amount"]),
    axis=1
)
#data.info()

data["employment_stability_score"] = data.apply(bm.employment_stability_score, axis=1)

data_depend_class = data["emi_eligibility"]
#data_depend_class
data_depend_reg = data["max_monthly_emi"]
#data_depend_reg
data = data.drop(["emi_eligibility", "max_monthly_emi"], axis=1)
data_independ = data

#distinct_genders = data_independ["gender"].unique()
#print(distinct_genders)
data_independ = pd.get_dummies(data_independ)
#data_independ.info()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


preprocessing_obj= bm.get_data_transformer_object(data_independ)


X_train,X_test,y_train,y_test = train_test_split(data_independ,data_depend_class,test_size=0.30,random_state=42)
X_train_reg,X_test_reg,y_train_reg,y_test_reg = train_test_split(data_independ,data_depend_reg,test_size=0.30,random_state=42)

#input_feature_train_arr=preprocessing_obj.fit_transform(X_train)
#input_feature_test_arr=preprocessing_obj.transform(X_test)
#input_feature_train_arr_reg=preprocessing_obj.fit_transform(X_train_reg)
#input_feature_test_arr_reg=preprocessing_obj.transform(X_test_reg)
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor



label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Example pipeline
numerical_columns = X_train.select_dtypes(include=['number']).columns

categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
])

best_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
    objective="multi:softprob",   # multiclass
    num_class=label_encoder.classes_,        # number of unique labels
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="mlogloss"        # multiclass log-loss
))
])
# -----------------------
# 5️⃣ Regression pipeline
# -----------------------
best_reg_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1))
])



#preprocessor.fit(X_train)
#input_feature_train_arr=preprocessor.fit_transform(X_train)

# Train and save
best_model.fit(X_train, y_train_encoded)
best_reg_model.fit(X_train_reg, y_train_reg)

# Save all
#joblib.dump(preprocessor, "scaler.pkl")
joblib.dump(best_model, "emi_model.pkl")
joblib.dump(best_reg_model, "emi_reg_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


label_encoder.fit(y_train)
preprocessing_obj.fit(X_train)

# Real sidebar (Streamlit's built-in one)
st.sidebar.title("Sidebar – Controls")
age = st.sidebar.text_input("Enter your age:")
gender_list = data["gender"].dropna().unique().tolist()
# Populate dropdown in sidebar (or main area)
selected_gender = st.sidebar.selectbox("Select Gender:", gender_list)

ms_list = data["marital_status"].dropna().unique().tolist()
# Populate dropdown in sidebar (or main area)
selected_ms = st.sidebar.selectbox("Select Marital Status:", ms_list)

education = data["education"].dropna().unique().tolist()
# Populate dropdown in sidebar (or main area)
selected_education = st.sidebar.selectbox("Select Education:", education)

monthly_salary = st.sidebar.text_input("Enter your Salary:")

emp_type = data["employment_type"].dropna().unique().tolist()
# Populate dropdown in sidebar (or main area)
selected_emtype = st.sidebar.selectbox("Select Employment Type:", emp_type)

YOE = st.sidebar.text_input("Enter your Years Of Employment:")

comp_type = data["company_type"].dropna().unique().tolist()
# Populate dropdown in sidebar (or main area)
selected_comp_type = st.sidebar.selectbox("Select Company TYpe:", comp_type)

house_type = data["house_type"].dropna().unique().tolist()
# Populate dropdown in sidebar (or main area)
selected_house_type = st.sidebar.selectbox("Select House TYpe:", house_type)

monthly_rent = st.sidebar.text_input("Enter your Monthly Rent:")

family_size = st.sidebar.text_input("Enter your Family Size:")

dependents = st.sidebar.text_input("Enter your Dependents:")

school_fees = st.sidebar.text_input("Enter your School Fee:")
college_fees = st.sidebar.text_input("Enter your College Fees:")
travel_expenses= st.sidebar.text_input("Enter your Travel Expenses:")

grocery_utility= st.sidebar.text_input("Enter your Grocery Utility:")
other_monthly_expenses= st.sidebar.text_input("Enter your Other Monthly Expenses:")

existing_loans = data["existing_loans"].dropna().unique().tolist()
# Populate dropdown in sidebar (or main area)
selected_existing_loan = st.sidebar.selectbox("Select Existing Loan:", existing_loans)

current_emi_amount = st.sidebar.text_input("Enter your Current EMI Amount:")

credit_score = st.sidebar.text_input("Enter your Credit Score:")

bank_balance = st.sidebar.text_input("Enter your Bank Balance:")

emegency_fund = st.sidebar.text_input("Enter your Emergency Fund:")

emi_scenario = data["emi_scenario"].dropna().unique().tolist()
# Populate dropdown in sidebar (or main area)
selected_emi_scenario = st.sidebar.selectbox("Select EMI Scenario:", emi_scenario)

requested_amount= st.sidebar.text_input("Enter your Requested Amount:")

requested_tenure= st.sidebar.text_input("Enter your Requested Tenure:")



if st.sidebar.button("Calculate EMI"):
    new_applicant = {
    "age": pd.to_numeric(age, errors="coerce"),
    "gender": selected_gender,
    "marital_status": selected_ms,
    "education": selected_education,
    "monthly_salary": pd.to_numeric(monthly_salary,errors="coerce"),
    "employment_type": selected_emtype,
    "years_of_employment": pd.to_numeric(YOE,errors="coerce"),
    "company_type": selected_comp_type,
    "house_type": selected_house_type,
    "monthly_rent": pd.to_numeric(monthly_rent,errors="coerce"),
    "family_size": pd.to_numeric(family_size,errors="coerce"),
    "dependents": pd.to_numeric(dependents,errors="coerce"),
    "school_fees": pd.to_numeric(school_fees,errors="coerce"),
    "college_fees": pd.to_numeric(college_fees,errors="coerce"),
    "travel_expenses": pd.to_numeric(travel_expenses,errors="coerce"),
    "groceries_utilities": pd.to_numeric(grocery_utility,errors="coerce"),
    "other_monthly_expenses": pd.to_numeric(other_monthly_expenses,errors="coerce"),
    "existing_loans": selected_existing_loan,
    "current_emi_amount": pd.to_numeric(current_emi_amount,errors="coerce"),
    "credit_score": pd.to_numeric(credit_score,errors="coerce"),
    "bank_balance": pd.to_numeric(bank_balance,errors="coerce"),
    "emergency_fund": pd.to_numeric(emegency_fund,errors="coerce"),
    "emi_scenario": emi_scenario,
    "requested_amount": pd.to_numeric(requested_amount,errors="coerce"),
    "requested_tenure": pd.to_numeric(requested_tenure,errors="coerce")
    }
# ✅ Convert to DataFrame
    new_data = pd.DataFrame([new_applicant])
    new_data["DTI"] = (new_data["current_emi_amount"]/new_data["monthly_salary"]).round(2)
    new_data["ETI"] = ((new_data["school_fees"]+new_data["college_fees"]+new_data["travel_expenses"]+new_data["groceries_utilities"]+new_data["other_monthly_expenses"])/new_data["monthly_salary"]).round(2)
    new_data["affordability ratios"] = ((new_data["current_emi_amount"]+new_data["school_fees"]+new_data["college_fees"]+new_data["travel_expenses"]+new_data["groceries_utilities"]+new_data["other_monthly_expenses"])/new_data["monthly_salary"]).round(2)

    new_data["emi_to_income_ratio"] = (new_data["current_emi_amount"]/new_data["monthly_salary"]).round(2)
    new_data["credit_risk_score"] = new_data.apply(
    lambda row: bm.calculate_credit_risk_score(row["credit_score"], row["current_emi_amount"]),
    axis=1
    )
    new_data["employment_stability_score"]=new_data.apply(bm.employment_stability_score, axis=1)

# Step 2️⃣ - Apply same preprocessing
# (Ensure you saved and reused preprocessing_obj)


    new_data = new_data.applymap(lambda x: str(x) if isinstance(x, list) else x)
    new_data = pd.get_dummies(new_data)
    missing_cols = set(X_train.columns) - set(new_data.columns)

    for col in missing_cols:
    # Assign default values based on column type
        if X_train[col].dtype == 'bool':
            new_data[col] = False

    print(new_data.info())

# Step 2️⃣ - Apply same preprocessing
# (Ensure you saved and reused preprocessing_obj)
# Drop the 'ID' column in-place
#X_train = X_train.drop(columns=['ID'])
    new_data = new_data[X_train.columns]
    print(new_data.shape)
#preprocessing_obj = get_data_transformer_object(new_data)
#preprocessing_obj.fit(new_data)

#processed_data = preprocessor.transform(new_data)

#print("Train shape:", X_train_processed.shape)
    
    #y_train_encoded_reg = label_encoder.fit_transform(y_train_reg)

    #scaler = joblib.load("scaler.pkl")
    best_model = joblib.load("emi_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    best_reg_model = joblib.load("emi_reg_model.pkl")

    # process new_applicant exactly as training
    #processed_data = scaler.transform(new_data)
    st.write("New row shape:", new_data.shape)

    predicted_encoded = best_model.predict(new_data)
    #predicted_encoded = (probs[:, 1] > 0.4).astype(int)

    # Step 4️⃣ - Decode label (e.g., 0 → Not Eligible, 1 → Eligible)
    predicted_label = label_encoder.inverse_transform(predicted_encoded)
    st.write(predicted_label)

    predicted_emi = best_reg_model.predict(new_data)

    st.write(f"\n🏦 EMI Eligibility Prediction: {predicted_label[0]}")
    st.write(f"💰 Predicted EMI Amount: ₹{predicted_emi[0]:,.2f}")

