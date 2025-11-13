# Define a class
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
class BusinessMethod:
    # Constructor method
    def __init__(self):
        self = self
        

    # Method
    @staticmethod
    def calculate_credit_risk_score(credit_score, current_emi_amount):
        # Normalize credit score (300–850)
        credit_score_norm = (credit_score - 300) / (850 - 300)
        credit_score_norm = max(0, min(credit_score_norm, 1))  # clamp 0–1

        # Normalize EMI amount (assuming 1L as high-risk threshold)
        emi_norm = current_emi_amount / 100000
        emi_norm = max(0, min(emi_norm, 1))  # clamp 0–1

        # Weighted score
        risk_score = (0.7 * credit_score_norm + 0.3 * (1 - emi_norm)) * 100
        return round(risk_score, 2)
    
    @staticmethod
    def employment_stability_score(row):
        # Employment type mapping
        emp_type_map = {
        "Government": 1.0,
        "Private": 0.75,
        "Self-employed": 0.6
        }

        f1 = emp_type_map.get(row["employment_type"], 0.5)
        f2 = min(row["years_of_employment"] / 10, 1)

        # Weighted formula
        score = 0.6 * f1 + 0.4 * f2
        return round(score, 3)
    
    @staticmethod
    def get_data_transformer_object(data_independ):
        '''
        This function is responsible for data trnasformation

        '''

        numerical_columns = data_independ.select_dtypes(include=['number']).columns
        numerical_columns = np.delete(numerical_columns, 0, axis=0)

        categorical_columns = data_independ.select_dtypes(include=['object', 'category']).columns
        num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )
        cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

        preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
        preprocess_type=type(preprocessor)
        return preprocessor

