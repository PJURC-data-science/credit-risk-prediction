import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import FunctionTransformer, Pipeline
from utils_app import create_features, clean_feature_name, identify_skewed_features
from utils_app import custom_box_cox_transform, inverse_box_cox_transform

import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
MODEL_NAME = 'TunedLightGBM.pkl'
COLUMNS_FILE = 'columns.txt'

# Load the model
model = joblib.load(MODEL_NAME)


def read_columns(cols_file):
    """Read the column names"""
    with open(cols_file, 'r') as file:
        columns = [line.strip() for line in file]

    return columns


def read_data(files):
    """Read the CSV files into DataFrames"""
    df = pd.read_csv(files['client'])
    df_bureau_balance = pd.read_csv(files['bureau_balance'])
    df_bureau = pd.read_csv(files['bureau'])
    df_previous_application = pd.read_csv(files['previous_application'])
    df_cc_balance = pd.read_csv(files['credit_card_balance'])
    df_installments_payments = pd.read_csv(files['installments_payments'])
    df_pos_cash_balance = pd.read_csv(files['pos_cash_balance'])

    return df, df_bureau_balance, df_bureau, df_previous_application, \
        df_cc_balance, df_installments_payments, df_pos_cash_balance


def encoding_step(X):
    """Encoding step"""
    # Encode categorical features
    categorical_columns = X.select_dtypes(exclude=['int64', 'float64']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

    encoded = pd.get_dummies(X[categorical_columns], drop_first=True)
    X = X[numerical_columns]
    X = X.join(encoded)

    # Clean feature names
    X.columns = X.columns.map(clean_feature_name)

    return X


def align_dataset(df):
    """Align the dataset with the original columns"""
    columns = read_columns(COLUMNS_FILE)
    df_aligned = pd.DataFrame(index=df.index, columns=columns)
    for col in columns:
        if col in df.columns:
            df_aligned[col] = df[col]
        else:
            df_aligned[col] = np.nan

    return df_aligned


def transform_and_scaling_step(X, skew_threshold=0.5, lmbda=0.15):

    # Identify numerical columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    skewed_features = identify_skewed_features(
        X[numerical_columns], threshold=skew_threshold)
    non_skewed_features = [
        col for col in numerical_columns if col not in skewed_features]

    # Create preprocessing pipelines
    skewed_transformer = Pipeline(
        steps=[
            ('box_cox', FunctionTransformer(
                lambda x: custom_box_cox_transform(
                    x, lmbda), inverse_func=lambda x: inverse_box_cox_transform(
                    x, lmbda))), ('scaler', StandardScaler())])
    non_skewed_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('skewed_num', skewed_transformer, skewed_features),
            ('non_skewed_num', non_skewed_transformer, non_skewed_features),
        ])

    # Fit the preprocessor
    preprocessor.fit(X)

    # Transform the data
    X_transformed = preprocessor.transform(X)

    # Get feature names after preprocessing
    skewed_feature_names = [clean_feature_name(
        f"{col}_box_cox") for col in skewed_features]
    non_skewed_feature_names = [
        clean_feature_name(col) for col in non_skewed_features]

    feature_names = skewed_feature_names + non_skewed_feature_names

    # Convert to DataFrame
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    return X_transformed


def classify_risk(probability):
    if probability > 0.7:
        return 'high'
    elif probability > 0.3:
        return 'moderate'
    else:
        return 'low'


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Server is running'})


@app.route('/predict', methods=['POST'])
def predict():
    # Read the CSV files into DataFrames
    df, df_bureau_balance, df_bureau, df_previous_application, \
        df_cc_balance, df_installments_payments, df_pos_cash_balance = read_data(request.files)

    # Create new features
    df = create_features(
        df,
        df_bureau_balance,
        df_bureau,
        df_cc_balance,
        df_installments_payments,
        df_pos_cash_balance,
        df_previous_application,
        is_train=False)

    # Client ID
    client_id = df['SK_ID_CURR']

    # Drop the client label
    df = df.drop(['SK_ID_CURR'], axis=1)

    # Encode
    df = encoding_step(df)

    # Transform and Scale
    df = transform_and_scaling_step(df)

    # Align the dataset to match model properties
    df = align_dataset(df)

    # Predict
    prediction_proba = model.predict_proba(df)
    default_probability = prediction_proba[:, 1]
    risk_scores = [classify_risk(proba) for proba in default_probability]

    return jsonify({
        'client_id': client_id.tolist(),
        'default_probability': default_probability.tolist(),
        'risk_scores': risk_scores
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
