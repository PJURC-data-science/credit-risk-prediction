import pandas as pd
import numpy as np
import re
from scipy.special import boxcox1p


def create_features(
        df: pd.DataFrame,
        df_bureau_balance: pd.DataFrame,
        df_bureau: pd.DataFrame,
        df_cc_balance: pd.DataFrame,
        df_installments_payments: pd.DataFrame,
        df_pos_cash_balance: pd.DataFrame,
        df_previous_application: pd.DataFrame,
        is_train: bool) -> pd.DataFrame:
    """
    Create features from the given dataframes.

    Parameters:
        df (pd.DataFrame): Main dataframe containing application data.
        df_bureau_balance (pd.DataFrame): Bureau balance dataframe.
        df_bureau (pd.DataFrame): Bureau dataframe.
        df_cc_balance (pd.DataFrame): Credit card balance dataframe.
        df_installments_payments (pd.DataFrame): Installments payment dataframe.
        df_pos_cash_balance (pd.DataFrame): Pos cash balance dataframe.
        df_previous_application (pd.DataFrame): Previous application dataframe.
        is_train (bool): Whether the data is for training or not.

    Returns:
        pd.DataFrame: The dataframe with the additional features.
    """
    # Debt-to-Income Ratio
    df['DEBT_TO_INCOME'] = df['AMT_CREDIT'] / \
        df['AMT_INCOME_TOTAL'].replace(0, np.nan)

    # Credit Utilization Rate
    cc_util = df_cc_balance.groupby('SK_ID_CURR').apply(
        lambda x: (
            x['AMT_BALANCE'] /
            x['AMT_CREDIT_LIMIT_ACTUAL'].replace(
                0,
                np.nan)).mean()).fillna(0)
    df = df.merge(
        cc_util.to_frame('CREDIT_UTILIZATION'),
        on='SK_ID_CURR',
        how='left')

    # Payment-to-Income Ratio
    df['PAYMENT_TO_INCOME'] = df['AMT_ANNUITY'] / \
        df['AMT_INCOME_TOTAL'].replace(0, np.nan)

    # Employment Stability
    df['EMPLOYMENT_STABILITY'] = pd.cut(
        df['DAYS_EMPLOYED'],
        bins=[-np.inf, -1825, -1095, -365, 0],
        labels=['>5y', '3-5y', '1-3y', '<1y']
    )

    # Age Group
    df['AGE_GROUP'] = pd.cut(
        -df['DAYS_BIRTH'] / 365,
        bins=[0, 25, 35, 45, 55, np.inf],
        labels=['<25', '25-35', '35-45', '45-55', '55+']
    )

    # Credit History Length
    credit_history = df_bureau.groupby(
        'SK_ID_CURR')['DAYS_CREDIT'].min().abs() / 365
    df = df.merge(
        credit_history.to_frame('CREDIT_HISTORY_YEARS'),
        on='SK_ID_CURR',
        how='left')

    # Recent Inquiry Intensity
    inquiry_columns = [
        col for col in df.columns if col.startswith('AMT_REQ_CREDIT_BUREAU_')]
    df['RECENT_INQUIRY_INTENSITY'] = df[inquiry_columns].sum(axis=1)

    # Asset Ownership Score
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
    df['ASSET_OWNERSHIP_SCORE'] = df['FLAG_OWN_CAR'].astype(
        int) + df['FLAG_OWN_REALTY'].astype(int)

    # Previous Loan Performance
    def _calc_bureau_performance(group):
        no_overdue = (group['AMT_CREDIT_SUM_OVERDUE'] == 0).mean()

        # Calculate credit utilization, handling division by zero
        credit_sum = group['AMT_CREDIT_SUM']
        credit_debt = group['AMT_CREDIT_SUM_DEBT']

        # Avoid division by zero by setting utilization to 1 when credit sum is
        # 0
        credit_utilization = np.where(
            credit_sum != 0, credit_debt / credit_sum, 1)

        # Calculate mean utilization, ignoring NaN values
        mean_utilization = np.nanmean(credit_utilization)

        # If all values were NaN, set mean_utilization to 1 (worst case)
        if np.isnan(mean_utilization):
            mean_utilization = 1

        # Combine the metrics to a performance score
        return 0.7 * no_overdue + 0.3 * (1 - mean_utilization)
    # Apply the function
    bureau_perform = df_bureau.groupby(
        'SK_ID_CURR').apply(_calc_bureau_performance)
    df = df.merge(
        bureau_perform.to_frame('PREV_LOAN_PERFORMANCE'),
        on='SK_ID_CURR',
        how='left')

    # Social Circle Default Rate
    df['SOCIAL_CIRCLE_DEFAULT_RATE'] = (
        df['DEF_30_CNT_SOCIAL_CIRCLE'] /
        df['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    ).fillna(0)

    # Credit Card Spending Behavior
    cc_spending = df_cc_balance.groupby('SK_ID_CURR').agg({
        'AMT_DRAWINGS_ATM_CURRENT': 'mean',
        'AMT_DRAWINGS_CURRENT': 'mean'
    })
    cc_spending['CC_SPENDING_RATIO'] = (
        cc_spending['AMT_DRAWINGS_ATM_CURRENT'] /
        cc_spending['AMT_DRAWINGS_CURRENT'].replace(0, np.nan)
    ).fillna(0)
    df = df.merge(cc_spending[['CC_SPENDING_RATIO']],
                  on='SK_ID_CURR', how='left')

    # Loan Application Frequency
    app_frequency = df_previous_application['SK_ID_CURR'].value_counts()
    df = df.merge(
        app_frequency.to_frame('LOAN_APPLICATION_FREQUENCY'),
        on='SK_ID_CURR',
        how='left')

    # Payment Timeliness
    payment_delay = df_installments_payments.groupby('SK_ID_CURR').apply(
        lambda x: (x['DAYS_ENTRY_PAYMENT'] - x['DAYS_INSTALMENT']).mean())
    df = df.merge(
        payment_delay.to_frame('AVG_PAYMENT_DELAY'),
        on='SK_ID_CURR',
        how='left')

    # Income to Credit Limit Ratio
    avg_credit_limit = df_cc_balance.groupby(
        'SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].mean().rename('AVG_CREDIT_LIMIT')
    df = df.merge(avg_credit_limit, on='SK_ID_CURR', how='left')
    df['INCOME_TO_CREDIT_LIMIT'] = df['AMT_INCOME_TOTAL'] / \
        df['AVG_CREDIT_LIMIT'].replace(0, np.nan)

    # Occupation Risk Score
    if is_train:
        occupation_risk = df.groupby('OCCUPATION_TYPE')[
            'TARGET'].mean().to_dict()
        pd.to_pickle(occupation_risk, 'dicts/occupation_risk.pkl')
    else:
        occupation_risk = pd.read_pickle('dicts/occupation_risk.pkl')

    df['OCCUPATION_RISK_SCORE'] = df['OCCUPATION_TYPE'].map(occupation_risk)
    df['OCCUPATION_RISK_SCORE'] = df['OCCUPATION_RISK_SCORE'].fillna(
        np.mean(list(occupation_risk.values())))

    # Region Risk Score
    if is_train:
        region_risk = df.groupby('REGION_RATING_CLIENT_W_CITY')[
            'TARGET'].mean().to_dict()
        pd.to_pickle(region_risk, 'dicts/region_risk.pkl')
    else:
        region_risk = pd.read_pickle('dicts/region_risk.pkl')

    df['REGION_RISK_SCORE'] = df['REGION_RATING_CLIENT_W_CITY'].map(
        region_risk)
    # Fill NaN values with the mean risk score
    df['REGION_RATING_CLIENT_W_CITY'] = df['REGION_RATING_CLIENT_W_CITY'].fillna(
        np.mean(list(region_risk.values())))

    # Cash Loan Proportion
    def _calculate_cash_loan_proportion(group):
        total_loans = len(group)
        cash_loans = (group['NAME_CONTRACT_TYPE'] == 'Cash loans').sum()
        return cash_loans / total_loans if total_loans > 0 else np.nan
    cash_loan_prop = df_previous_application.groupby(
        'SK_ID_CURR').apply(_calculate_cash_loan_proportion)
    df = df.merge(
        cash_loan_prop.to_frame('CASH_LOAN_PROPORTION'),
        on='SK_ID_CURR',
        how='left')

    # Behavioral Score
    df['BEHAVIORAL_SCORE'] = (
        df['OCCUPATION_RISK_SCORE'].fillna(df['OCCUPATION_RISK_SCORE'].mean()) * 0.2 +
        df['REGION_RISK_SCORE'].fillna(df['REGION_RISK_SCORE'].mean()) * 0.2 +
        df['PREV_LOAN_PERFORMANCE'].fillna(df['PREV_LOAN_PERFORMANCE'].mean()) * 0.3 +
        (1 - df['CREDIT_UTILIZATION'].fillna(df['CREDIT_UTILIZATION'].mean())) * 0.15 +
        (1 - df['DEBT_TO_INCOME'].fillna(df['DEBT_TO_INCOME'].mean())) * 0.15
    )
    # Normalize the BEHAVIORAL_SCORE to be between 0 and 1
    min_score = df['BEHAVIORAL_SCORE'].min()
    max_score = df['BEHAVIORAL_SCORE'].max()
    df['BEHAVIORAL_SCORE'] = (
        df['BEHAVIORAL_SCORE'] - min_score) / (max_score - min_score)

    # Average Days Past Due (DPD) Ratio
    def _calculate_dpd_ratio(group):
        total_records = len(group)
        dpd_records = (group['SK_DPD'] > 0).sum()
        return dpd_records / total_records if total_records > 0 else 0
    pos_cash_dpd_ratio = df_pos_cash_balance.groupby(
        'SK_ID_CURR').apply(_calculate_dpd_ratio)
    df = df.merge(pos_cash_dpd_ratio.to_frame(
        'POS_CASH_DPD_RATIO'), on='SK_ID_CURR', how='left')

    # Trend in number of future installments
    def _calculate_installment_trend(group):
        if len(group) < 2:
            return 0
        group = group.sort_values('MONTHS_BALANCE')
        first_half = group['CNT_INSTALMENT_FUTURE'].iloc[:len(
            group) // 2].mean()
        second_half = group['CNT_INSTALMENT_FUTURE'].iloc[len(
            group) // 2:].mean()
        return second_half - first_half
    pos_cash_installment_trend = df_pos_cash_balance.groupby(
        'SK_ID_CURR').apply(_calculate_installment_trend)
    df = df.merge(pos_cash_installment_trend.to_frame(
        'POS_CASH_INSTALLMENT_TREND'), on='SK_ID_CURR', how='left')

    # Proportion of 'Active' loans
    def _calculate_active_loans_ratio(group):
        total_records = len(group)
        # 'C' typically means 'Closed'
        active_records = (group['STATUS'] == 'C').sum()
        return 1 - (active_records / total_records) if total_records > 0 else 0
    bureau_active_ratio = df_bureau_balance.groupby(
        'SK_ID_BUREAU').apply(_calculate_active_loans_ratio)
    df_bureau = df_bureau.merge(bureau_active_ratio.to_frame(
        'BUREAU_ACTIVE_LOANS_RATIO'), on='SK_ID_BUREAU', how='left')
    bureau_active_mean = df_bureau.groupby(
        'SK_ID_CURR')['BUREAU_ACTIVE_LOANS_RATIO'].mean()
    df = df.merge(
        bureau_active_mean.to_frame('BUREAU_ACTIVE_LOANS_RATIO_MEAN'),
        on='SK_ID_CURR',
        how='left')

    return df


def clean_feature_name(name: str) -> str:
    """
    Replace non-alphanumeric characters with underscores in a given string.

    Args:
        name (str): The string to clean.

    Returns:
        str: The cleaned string.
    """
    return re.sub(r'[^\w]+', '_', name)


def identify_skewed_features(df: pd.DataFrame, threshold: float = 0.5) -> list:
    """
    Identifies numerical features with skewness above a given threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float, optional): The skewness threshold above which a feature is considered skewed. Defaults to 0.5.

    Returns:
        list: A list of feature names with skewness above the threshold.
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    skewness = df[numeric_features].skew()
    skewed_features = skewness[abs(skewness) > threshold].index
    return list(skewed_features)


def custom_box_cox_transform(X: np.ndarray, lmbda: float = 0.15) -> np.ndarray:
    """
    Applies the Box-Cox transformation to positive values in a given array.

    Args:
        X (np.ndarray): The input array.
        lmbda (float, optional): The lambda parameter for the Box-Cox transformation. Defaults to 0.15.

    Returns:
        np.ndarray: The transformed array.
    """
    return np.where(X > 0, boxcox1p(X, lmbda), X)


def inverse_box_cox_transform(
        X: np.ndarray,
        lmbda: float = 0.15) -> np.ndarray:
    """
    Applies the inverse Box-Cox transformation to an array.

    Args:
        X (np.ndarray): The input array.
        lmbda (float, optional): The lambda parameter for the inverse Box-Cox transformation. Defaults to 0.15.

    Returns:
        np.ndarray: The inverse-transformed array.
    """
    return np.where(X > 0, np.expm1(X) if lmbda ==
                    0 else (X * lmbda + 1)**(1 / lmbda) - 1, X)
