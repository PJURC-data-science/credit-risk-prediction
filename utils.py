import re
from typing import Dict, List, Tuple
from category_encoders import MEstimateEncoder, OneHotEncoder
from scipy import stats
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import ceil
import phik
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, confusion_matrix, log_loss, precision_recall_curve, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 98

COLOR_PALETTE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]
EXPORT_FOLDER = 'exports'


def export_predictions(
        model: object,
        X_test: pd.DataFrame,
        test_id: pd.Series,
        submission_name: str) -> None:
    """Export model predictions to a .csv file"""
    # Predict probabilities
    probs = model.predict_proba(X_test)
    positive_probs = probs[:, 1]
    output = pd.DataFrame({'SK_ID_CURR': test_id, 'TARGET': positive_probs})
    output.to_csv(
        f'{EXPORT_FOLDER}/submission_{submission_name}.csv',
        index=False)


def custom_format(x: float) -> str:
    """
    Formats a given number to a string with a specific decimal precision.

    Args:
        x (float): The number to be formatted.

    Returns:
        str: The formatted number as a string. If the number is an integer, it is formatted as an integer with no decimal places.
        Otherwise, it is formatted with two decimal places.
    """
    if x == int(x):
        return '{:.0f}'.format(x)
    else:
        return '{:.2f}'.format(x)


def exclude_list_value(cols_list: list, value: str) -> list:
    """Returns a list of columns excluding the given value."""
    cols_list = [col for col in cols_list if col != value]

    return cols_list


def get_screen_width() -> int:
    """Retrieves the screen width using a tkinter root window and returns the screen width value."""
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    root.destroy()

    return screen_width


def set_font_size() -> dict:
    """Sets the font sizes for visualization elements based on the screen width."""
    base_font_size = round(get_screen_width() / 100, 0)
    font_sizes = {
        'font.size': base_font_size * 0.6,
        'axes.titlesize': base_font_size * 0.4,
        'axes.labelsize': base_font_size * 0.6,
        'xtick.labelsize': base_font_size * 0.4,
        'ytick.labelsize': base_font_size * 0.4,
        'legend.fontsize': base_font_size * 0.6,
        'figure.titlesize': base_font_size * 0.6
    }

    return font_sizes


def check_duplicates(df: pd.DataFrame, df_name: str) -> None:
    """
    Check for duplicate rows in a pandas DataFrame and print the results.

    Args:
        df (pandas.DataFrame): The DataFrame to check for duplicates.
        df_name (str): The name of the DataFrame for printing purposes.

    Returns:
        None
    """
    duplicate_count = df.duplicated().sum()
    print(f"DataFrame: {df_name}")
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows: {duplicate_count}\n")
    duplicates = df[df.duplicated(keep=False)]
    sorted_duplicates = duplicates.sort_values(by=list(df.columns))
    sorted_duplicates[:10] if len(duplicates) > 0 else None


def check_feature_importance_correlation(
        df: pd.DataFrame,
        target_column: str) -> pd.Series:
    """
    Checks the correlation between feature importance and the target column in a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features and target column.
        target_column (str): The name of the target column in the DataFrame.

    Returns:
        pd.Series: A Series containing the feature importances sorted in descending order.
    """

    # Prepare the data
    categorical_columns = df.select_dtypes(
        include=['object', 'category']).columns
    X = pd.get_dummies(
        df.drop(
            columns=[target_column]),
        columns=categorical_columns)
    y = df[target_column]

    # Simple imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_imputed, y)

    # Get feature importances
    importances = pd.Series(
        rf.feature_importances_,
        index=X.columns).sort_values(
        ascending=False)

    return importances


def train_test_missing_values(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        missing_only=False) -> pd.DataFrame:
    """Returns a DataFrame that summarizes missing values in the train and test datasets, including column data types, sorted by column data type."""
    missing_values_train = round(df_train.isnull().sum(), 0)
    missing_values_perc_train = round(
        (missing_values_train / len(df_train)) * 100, 1)
    missing_values_test = round(df_test.isnull().sum(), 0)
    missing_values_perc_test = round(
        (missing_values_test / len(df_test)) * 100, 1)
    column_data_types = df_train.dtypes

    missing_values = pd.DataFrame({
        'Data Type': column_data_types,
        'Train #': missing_values_train,
        'Train %': missing_values_perc_train,
        'Test #': missing_values_test,
        'Test %': missing_values_perc_test
    })

    missing_values = missing_values.sort_values(by='Train %', ascending=False)

    # Filter features with missing values
    if missing_only:
        missing_values = missing_values[
            (missing_values['Train #'] > 0) |
            (missing_values['Test #'] > 0)]

    return missing_values


def draw_predictor_numerical_plots(
        df: pd.DataFrame,
        predictor: str,
        target: str,
        hist_type='histogram') -> None:
    """
    Draws two plots to visualize the frequency counts and box plot of the distribution of a predictor variable by a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        predictor (str): The name of the predictor variable.
        target (str): The name of the target variable.
        hist_type (str): The type of plot to draw. Can be 'histogram' or 'kde'. Defaults to 'histogram'.

    Returns:
        None
    """
    fig_width = get_screen_width() / 100
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(fig_width, fig_width / 5))

    # Chart 1: Box Plot
    sns.boxplot(
        data=df,
        x=target,
        y=predictor,
        hue=target,
        palette=COLOR_PALETTE,
        saturation=0.75,
        legend=False,
        ax=ax1)
    ax1.set_title(f'Distribution of {predictor} by {target}')
    ax1.set_xlabel(f'{target}')
    ax1.set_ylabel(f'{predictor}')

    # Chart 2: Histogram
    if hist_type == 'kde':
        sns.kdeplot(
            data=df,
            x=predictor,
            hue=target,
            multiple='stack',
            palette=COLOR_PALETTE,
            ax=ax2)
    else:
        sns.histplot(
            data=df,
            x=predictor,
            hue=target,
            multiple='stack',
            palette=COLOR_PALETTE,
            ax=ax2)
    ax2.set_title(
        f'Frequency Distribution of {predictor.title()} by {target.title()}')
    ax2.set_xlabel(f'{predictor.title()}')
    ax2.set_ylabel('Count')

    plt.show()
    plt.close(fig)


def numerical_predictor_significance_test(
        df: pd.DataFrame,
        predictor: str,
        target: str,
        test_type='mann_whitney',
        missing_strategy='drop',
        min_sample_size=30) -> dict:
    """
    Perform either Mann-Whitney U test or Mood's median test and return the results.

    Args:
    df (pandas.DataFrame): The dataframe containing the data
    predictor (str): The name of the column containing the numerical predictor
    target (str): The name of the column containing the binary target
    test_type (str): Either 'mann_whitney' or 'moods_median'. Default is 'mann_whitney'
    missing_strategy (str): How to handle missing values. Options: 'drop', 'median_impute'
    min_sample_size (int): Minimum sample size required for each group

    Returns:
    dict: A dictionary containing the test results
    """
    # Handle missing values
    if missing_strategy == 'drop':
        df = df.dropna(subset=[predictor, target])
    elif missing_strategy == 'median_impute':
        df[predictor] = df[predictor].fillna(df[predictor].median())
    else:
        raise ValueError(
            "Invalid missing_strategy. Choose 'drop' or 'median_impute'")

    # Separate the data into two groups based on the binary target
    group1 = df[df[target] == 0][predictor]
    group2 = df[df[target] == 1][predictor]

    # Check if there's enough data
    if len(group1) < min_sample_size or len(group2) < min_sample_size:
        return {
            'error': f'Insufficient data. Group sizes: {
                len(group1)}, {
                len(group2)}'}

    if test_type == 'mann_whitney':
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
        effect_size = 2 * statistic / \
            (len(group1) * len(group2)) - 1  # Cliff's delta
    elif test_type == 'moods_median':
        statistic, p_value, _, _ = stats.median_test(group1, group2)
        test_name = "Mood's median test"
        effect_size = (np.mean(group1) - np.mean(group2)) / np.sqrt(
            (np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)  # Cohen's d
    else:
        raise ValueError(
            "Invalid test_type. Choose either 'mann_whitney' or 'moods_median'")

    results = {
        'test_name': test_name,
        'p_value': p_value,
        'statistic': statistic,
        'effect_size': effect_size,
        'group1_median': np.median(group1),
        'group2_median': np.median(group2),
        'group1_size': len(group1),
        'group2_size': len(group2)
    }

    return results


def interpret_results_numerical(
        df: pd.DataFrame,
        results: dict,
        col_name: str) -> pd.DataFrame:
    """Interpret the results of the non-parametric test and store them in a DataFrame"""
    data = {
        'Column': col_name,
        'Test Name': [results['test_name']],
        'P-value': [round(results['p_value'], 6)],
        'Test Statistic': [round(results['statistic'], 2)],
        'Effect Size': [round(results['effect_size'], 4)],
        'Median Group 0': [results['group1_median']],
        'Median Group 1': [results['group2_median']],
        'Significance': ['Statistically significant' if results['p_value'] < 0.05 else 'Not statistically significant'],
        'Effect Magnitude': []
    }

    if results['test_name'] == "Mann-Whitney U test":
        if abs(results['effect_size']) < 0.2:
            effect_magnitude = "negligible"
        elif abs(results['effect_size']) < 0.5:
            effect_magnitude = "small"
        elif abs(results['effect_size']) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
    else:  # Mood's median test
        if abs(results['effect_size']) < 0.2:
            effect_magnitude = "small"
        elif abs(results['effect_size']) < 0.5:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

    data['Effect Magnitude'].append(effect_magnitude)
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    return df


def draw_predictor_categorical_plots(
        df: pd.DataFrame,
        predictor: str,
        target: str) -> None:
    """
    Draws two plots to visualize the frequency counts and proportions of a predictor variable by a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        predictor (str): The name of the predictor variable.
        target (str): The name of the target variable.

    Returns:
        None
    """
    fig_width = get_screen_width() / 100
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(fig_width, fig_width / 5))

    # Ensure predictor and target is treated as categorical
    df[predictor] = df[predictor].astype('category')
    df[target] = df[target].astype('category')

    # Chart 1: Frequencies
    sns.countplot(
        data=df,
        x=predictor,
        hue=target,
        palette=COLOR_PALETTE,
        ax=ax1
    )
    ax1.set_title(f'Frequency Counts of {predictor} by {target}')
    ax1.set_xlabel(f'{predictor}')
    ax1.set_ylabel('Count')

    # Chart 2: Proportions
    props = df.groupby(predictor)[target].value_counts(normalize=True).unstack().reset_index().melt(id_vars=predictor)
    sns.barplot(
        data=props,
        x=predictor,
        y='value',
        hue=target,
        palette=COLOR_PALETTE,
        ax=ax2
    )
    ax2.set_title(f'Proportion of {target} by {predictor}')
    ax2.set_xlabel(f'{predictor}')
    ax2.set_ylabel('Proportion')
    ax2.legend().set_visible(False)
    ax2.tick_params(axis='x', rotation=0)

    plt.show()
    plt.close(fig)


def categorical_predictor_significance_test(
        df: pd.DataFrame,
        predictor: str,
        target: str,
        missing_strategy='drop') -> dict:
    """
    Performs chi-squared test for independence between a categorical predictor and binary target.

    Args:
    df (pandas.DataFrame): The dataframe containing the data
    predictor (str): The name of the column containing the categorical predictor
    target (str): The name of the column containing the binary target
    missing_strategy (str): How to handle missing values. Options: 'drop', 'most_frequent'

    Returns:
    dict: A dictionary containing the test results
    """
    # Handle missing values
    if missing_strategy == 'drop':
        df = df.dropna(subset=[predictor, target])
    elif missing_strategy == 'most_frequent':
        df[predictor] = df[predictor].fillna(df[predictor].mode()[0])
    else:
        raise ValueError(
            "Invalid missing_strategy. Choose 'drop' or 'most_frequent'")

    # Create a contingency table
    contingency_table = pd.crosstab(df[predictor], df[target])

    # Perform chi-squared test
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table)

    # Calculate Cramer's V for effect size
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))

    results = {
        'test_name': "Chi-squared test",
        'p_value': p_value,
        'chi2_statistic': chi2,
        'degrees_of_freedom': dof,
        'effect_size': cramer_v,
        'contingency_table': contingency_table
    }

    return results


def interpret_results_categorical(
        df: pd.DataFrame,
        results: dict,
        col_name: str) -> pd.DataFrame:
    """Interpret the results of the chi-squared test. Store the summary in a DataFrame"""
    data = {
        'Column': col_name,
        'Test Name': [results['test_name']],
        'P-value': [round(results['p_value'], 6)],
        'Chi-squared statistic': [round(results['chi2_statistic'], 2)],
        'Degrees of freedom': [round(results['degrees_of_freedom'], 4)],
        'Effect size (Cramer\'s V)': [round(results['effect_size'], 4)],
        'Significance': ['Statistically significant' if results['p_value'] < 0.05 else 'Not statistically significant'],
        'Effect Magnitude': []
    }

    # Interpret effect size (Cramer's V)
    if results['effect_size'] < 0.1:
        effect_magnitude = "negligible"
    elif results['effect_size'] < 0.3:
        effect_magnitude = "small"
    elif results['effect_size'] < 0.5:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"
    data['Effect Magnitude'] = effect_magnitude

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    return df


def phik_matrix(df: pd.DataFrame, numerical_columns: list,
                target_column: str = 'TARGET') -> tuple:
    """
    Calculates the Phi_k correlation coefficient matrix for the given DataFrame and columns,
    and returns the top 10 largest phik coefficients between the target feature and other features,
    as well as the top 10 interactions between any features.

    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_columns (list): List of numerical columns.
        target_column (str): Name of the target column. Defaults to 'TARGET'.

    Returns:
        tuple: (DataFrame of top 10 target correlations, DataFrame of top 10 overall interactions)
    """
    # Calculate Phi_k correlation matrix
    corr_matrix = df.phik_matrix(interval_cols=numerical_columns)

    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(get_screen_width() / 100 * 0.8, 8))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5)
    plt.title('Phi_k Correlation Heatmap')
    plt.show()

    # Extract correlations with the target feature
    target_correlations = corr_matrix[target_column].sort_values(
        ascending=False)

    # Remove self-correlation (correlation of TARGET with itself)
    target_correlations = target_correlations[target_correlations.index != target_column]

    # Get top 10 correlations with target
    top_10_target = target_correlations.head(10)

    # Create a DataFrame with feature names and their correlations to target
    target_df = pd.DataFrame({
        'Feature': top_10_target.index,
        'Phik Coefficient': top_10_target.values
    })

    # Get top 10 overall interactions
    # Create a DataFrame from the correlation matrix
    corr_df = corr_matrix.unstack().reset_index()
    corr_df.columns = ['Feature1', 'Feature2', 'Phik Coefficient']

    # Remove self-correlations and duplicate pairs
    corr_df = corr_df[corr_df['Feature1'] < corr_df['Feature2']]

    # Sort by absolute correlation and get top 10
    top_10_interactions = corr_df.sort_values(
        'Phik Coefficient', key=abs, ascending=False).head(10)

    return target_df, top_10_interactions


def draw_original_log_distribution(df: pd.DataFrame, feature: str) -> None:
    """
    Draws two plots to visualize the distributions of a feature variable and the log-transformed distribution.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        feature (str): The name of the feature to visualize.

    Returns:
        None
    """

    fig_width = get_screen_width() / 100
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(fig_width, fig_width / 5))

    # Chart 1: Original distribution
    df[feature].hist(color=COLOR_PALETTE[0], grid=False, ax=ax1)
    ax1.set_title(f'Original distribution of {feature}')
    ax1.set_xlabel(f'{feature}')
    ax1.set_ylabel('Count')

    # Chart 2: Log-transformed distribution
    np.log1p(df[feature]).hist(color=COLOR_PALETTE[1], grid=False, ax=ax2)
    ax2.set_title(f'Log Distribution of {feature}')
    ax2.set_xlabel(f'{feature}')
    ax2.set_ylabel('Count')

    plt.show()
    plt.close(fig)


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
    df['OCCUPATION_RISK_SCORE'].fillna(
        np.mean(list(occupation_risk.values())), inplace=True)

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
    df['REGION_RATING_CLIENT_W_CITY'].fillna(
        np.mean(list(region_risk.values())), inplace=True)

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


def categorize_features(df: pd.DataFrame,
                        categorical_features: list,
                        cardinality_threshold: float = 10) -> Tuple[list,
                                                                    list]:
    """
    Categorizes categorical features into high and low cardinality.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): The list of categorical features to categorize.
        cardinality_threshold (float, optional): The threshold for classifying a feature as high or low cardinality. Defaults to 10.

    Returns:
        list, list: Two lists of feature names, one for high cardinality features and one for low cardinality features.
    """
    high_cardinality = []
    low_cardinality = []
    for feature in categorical_features:
        if df[feature].nunique() > cardinality_threshold:
            high_cardinality.append(feature)
        else:
            low_cardinality.append(feature)
    return high_cardinality, low_cardinality


def calculate_roc_auc_score(model: object,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            rskf: RepeatedStratifiedKFold,
                            n_repeats: int = 3,
                            print_scores: bool = True) -> Tuple[float,
                                                                np.ndarray,
                                                                np.ndarray]:
    # Convert DataFrame to numpy array
    """
    Calculates the ROC AUC score of a model using cross-validation.

    Args:
        model (object): The model to evaluate.
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training labels.
        rskf (RepeatedStratifiedKFold): The cross-validation object.
        n_repeats (int, optional): The number of repeats. Defaults to 3.
        print_scores (bool, optional): Whether to print the scores. Defaults to True.

    Returns:
        List[float, np.ndarray, np.ndarray]: A list containing the overall ROC AUC, the predicted labels, and the predicted probabilities.
    """
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    # Ensure y_train is 1D
    y_train = y_train.ravel()

    # Initialize arrays to store predictions and fold scores
    y_pred = np.zeros_like(y_train, dtype=float)
    y_pred_proba = np.zeros((len(y_train), 2), dtype=float)
    fold_scores = []

    # Perform cross-validation manually
    for _, (train_index, test_index) in enumerate(
            rskf.split(X_train, y_train), 1):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # Fit the model and make predictions
        model.fit(X_train_fold, y_train_fold)
        fold_pred_proba = model.predict_proba(X_test_fold)
        fold_pred = fold_pred_proba[:, 1]

        # Update overall predictions
        y_pred[test_index] += fold_pred
        y_pred_proba[test_index] += fold_pred_proba

        # Calculate fold-specific ROC AUC
        fpr, tpr, _ = roc_curve(y_test_fold, fold_pred)
        fold_auc = auc(fpr, tpr)
        fold_scores.append(fold_auc)

    # Average the predictions
    y_pred /= n_repeats
    y_pred_proba /= n_repeats

    # Calculate overall ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_train, y_pred)
    roc_auc = auc(fpr, tpr)

    # Calculate stability metrics
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    ci_lower, ci_upper = stats.t.interval(
        0.95, len(fold_scores) - 1, loc=mean_score, scale=stats.sem(fold_scores))

    if print_scores:
        print(f"\nModel: {type(model).__name__}")
        print(f"Overall ROC AUC: {roc_auc:.4f}")
        print(f"Mean Fold ROC AUC: {mean_score:.4f} (+/- {std_score:.4f})")
        print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

        # Calculate coefficient of variation as a stability measure
        cv = std_score / mean_score
        print(f"Coefficient of Variation: {cv:.4f}")

        if cv < 0.05:
            print("Model performance is very stable across folds.")
        elif cv < 0.1:
            print("Model performance is reasonably stable across folds.")
        else:
            print("Model performance shows significant variation across folds.")

    # Return predictions, probability predictions, and fold scores
    y_pred = (y_pred > 0.5).astype(int)
    y_pred_proba = y_pred_proba[:, 1]
    return roc_auc, y_pred, y_pred_proba


def calculate_class_weights(
        y: pd.Series) -> Tuple[Dict[int, float], np.ndarray, float]:
    """
    Calculate class weights based on the distribution in y.

    :param y: Array-like, target variable
    :return: Dictionary of class weights, array of class weights, and the weight ratio
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    # Calculate the ratio of the majority class to the minority class
    weight_ratio = max(weights) / min(weights)

    return class_weights, weights, weight_ratio


def visualize_performance(y: pd.Series, y_pred: pd.Series,
                          y_pred_proba: pd.Series, model_name: str) -> None:
    """
    Visualizes the model performance

    Args:
        y (ndarray): True labels
        y_pred (ndarray): Predicted labels
        y_pred_proba (ndarray): Predicted probabilities
        model_name (str): Name of the model

    Returns:
        None
    """
    # 3 subplots
    _, ax = plt.subplots(1, 3, figsize=(16, 6))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    cm_normalized = cm / cm.sum()
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax[0])
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    ax[0].set_title(f'Confusion Matrix - {model_name}')

    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)
    ax[1].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], 'k--')  # Diagonal line
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver Operating Characteristic (ROC) Curve')
    ax[1].legend(loc='lower right')

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    ax[2].plot(recall, precision)
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_title('Precision-Recall Curve')

    plt.tight_layout()
    plt.show()


def get_feature_importance(
        model: object,
        feature_names: List[str]) -> pd.DataFrame:
    # Initialize a dictionary to store combined feature importances
    """
    Combine feature importances from multiple models (e.g. Random Forest, XGBoost, LightGBM) into a single DataFrame.

    Parameters:
    model (object): The model object (either a single model or an ensemble)
    feature_names (List[str]): The names of the features

    Returns:
    pd.DataFrame: A DataFrame containing the combined feature importances, sorted in descending order
    """

    combined_importances = {feature: 0 for feature in feature_names}

    # Check if the model is an ensemble
    if hasattr(model, 'estimators_'):
        # Get the constituent models
        models = [
            estimator for _,
            estimator in model.named_estimators_.items()]
    else:
        # Treat the model as a single model
        models = [model]

    for model in models:
        if hasattr(model, 'feature_importances_'):
            # For tree-based models (Random Forest, XGBoost, LightGBM)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = np.abs(model.coef_)
        else:
            print(
                f"Warning: {
                    type(model).__name__} doesn't have a standard feature importance attribute.")
            continue

        # Add importance to the combined importances
        for feature, importance in zip(feature_names, importances):
            combined_importances[feature] += importance

    # Average the importances
    for feature in combined_importances:
        combined_importances[feature] /= len(models)

    # Convert to DataFrame and sort
    importance_df = pd.DataFrame.from_dict(
        combined_importances,
        orient='index',
        columns=['Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False)

    return importance_df


def plot_probability_distribution(
        y_true: pd.Series, y_proba_preds: Dict[str, pd.Series]) -> None:
    """
    Plots the probability distribution of the predicted probabilities for each class.

    Args:
        y_true (pd.Series): The true labels of the data.
        y_proba_preds (Dict[str, pd.Series]): A dictionary containing the predicted probabilities for each class, with the
            model name as the key and the predicted probabilities as the value.

    Returns:
        None
    """
    _, ax = plt.subplots(1, 2, figsize=(16, 6))

    for i, (model_name, y_pred_proba) in enumerate(y_proba_preds.items()):
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Probability': y_pred_proba,
            'True_Class': y_true
        })

        # Plot KDE for each class
        sns.kdeplot(data=df[df['True_Class'] == 0],
                    x='Probability',
                    shade=True,
                    color="skyblue",
                    label="Class 0 (Negative)",
                    ax=ax[i])
        sns.kdeplot(data=df[df['True_Class'] == 1],
                    x='Probability',
                    shade=True,
                    color="orange",
                    label="Class 1 (Positive)",
                    ax=ax[i])

        # Add vertical lines for potential thresholds
        ax[i].axvline(
            x=0.5,
            color='red',
            linestyle='--',
            label='Default Threshold (0.5)')

        # Customize the plot
        ax[i].set_title(
            f'Distribution of Prediction Probabilities - {model_name}')
        ax[i].set_xlabel('Predicted Probability of Positive Class')
        ax[i].set_ylabel('Density')
        ax[i].legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def ensemble_feature_importance(
        X: pd.DataFrame,
        y: pd.Series,
        n_iterations: int = 5) -> pd.Series:
    """
    Calculate the ensemble feature importance by training multiple models and averaging their feature importances.

    Parameters:
        X (pd.DataFrame): The input DataFrame containing the features.
        y (pd.Series): The target variable.
        n_iterations (int): The number of times to train each model. Defaults to 5.

    Returns:
        pd.Series: The ensemble feature importance, with the feature names as the index and the importance values as the values.
    """
    class_weights, _, weight_ratio = calculate_class_weights(y)
    models = {
        'xgboost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='auc',
            n_jobs=-1,
            scale_pos_weight=weight_ratio,
            enable_categorical=True),
        'lightgbm': LGBMClassifier(
            n_jobs=-1,
            class_weight=class_weights,
            verbose=-1),
    }

    feature_importance_sum = {model: pd.Series(
        0, index=X.columns) for model in models}

    # Identify categorical columns
    categorical_columns = X.select_dtypes(exclude=['int64', 'float64']).columns

    # Convert categorical columns to 'category' dtype
    for col in categorical_columns:
        X[col] = X[col].astype('category')

    # Cleanup column names
    X.columns = [clean_feature_name(name) for name in X.columns]

    for i in range(n_iterations):
        X_split, _, y_split, _ = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE + i)

        for model_name, model in models.items():
            model.random_state = RANDOM_STATE + i

            if model_name == 'xgboost':
                model.fit(X_split, y_split)
            elif model_name == 'lightgbm':
                model.fit(
                    X_split,
                    y_split,
                    categorical_feature=categorical_columns.tolist())

            importance = pd.Series(model.feature_importances_, index=X.columns)

            feature_importance_sum[model_name] += importance

    # Calculate average importance for each model
    average_importance = {
        model: importance_sum /
        n_iterations for model,
        importance_sum in feature_importance_sum.items()}

    # Calculate ensemble average importance
    ensemble_average = pd.DataFrame(average_importance).mean(axis=1)

    return ensemble_average


def weighted_average_ensemble(
        weights: List[float],
        *predictions: np.ndarray) -> np.ndarray:
    """
    Calculate the weighted average of multiple predictions.

    Parameters:
        weights (List[float]): A list of weights corresponding to each prediction.
        *predictions (List[np.ndarray]): A list of predictions to be averaged.

    Returns:
        np.ndarray: The weighted average of the predictions.
    """
    return np.average(np.array(predictions), axis=0, weights=weights)


def negative_auc_roc(weights: List[float], *args: np.ndarray) -> float:
    """
    Calculate the negative AUC-ROC score (for minimization).

    Parameters:
        weights (List[float]): A list of weights corresponding to each prediction.
        *args: A variable number of arguments, where the first argument is the true labels (y_true)
            and the remaining arguments are predicted probabilities.

    Returns:
        float: The negative AUC-ROC score.
    """
    y_true = args[0]
    predictions = args[1:]
    final_prediction = weighted_average_ensemble(weights, *predictions)

    return -roc_auc_score(y_true, final_prediction)


def encoding_step(X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  cardinality_threshold: int = 10) -> Tuple[pd.DataFrame,
                                                            pd.DataFrame,
                                                            pd.Series]:
    """
    Encode categorical features of the training and test datasets.

    The function splits the categorical features into two groups: low cardinality features and high cardinality features.
    Low cardinality features are one-hot encoded, while high cardinality features are mean encoded.

    Parameters:
        X_train (pd.DataFrame): The training dataset.
        X_test (pd.DataFrame): The test dataset.
        y_train (pd.Series): The target variable for the training dataset.
        cardinality_threshold (int, optional): The threshold for classifying a feature as low or high cardinality. Defaults to 10.

    Returns:
        X_train_preprocessed (pd.DataFrame): The preprocessed training dataset.
        X_test_preprocessed (pd.DataFrame): The preprocessed test dataset.
        y_train (pd.Series): The target variable for the training dataset.
    """

    categorical_features = X_train.select_dtypes(
        exclude=['int64', 'float64']).columns
    high_cardinality, low_cardinality = categorize_features(
        X_train, categorical_features, cardinality_threshold)

    low_card_transformer = OneHotEncoder(handle_unknown='ignore')
    low_card_encoded = low_card_transformer.fit_transform(
        X_train[low_cardinality])
    low_card_feature_names = [
        clean_feature_name(name) for name in
        low_card_transformer.get_feature_names_out(low_cardinality)
    ]
    X_train_low_card = pd.DataFrame(
        low_card_encoded,
        columns=low_card_feature_names,
        index=X_train.index)
    X_test_low_card = pd.DataFrame(
        low_card_transformer.transform(X_test[low_cardinality]),
        columns=low_card_feature_names,
        index=X_test.index
    )

    # Mean encoding for high cardinality features
    high_card_transformer = MEstimateEncoder(cols=high_cardinality, m=1)
    X_train_high_card = high_card_transformer.fit_transform(
        X_train[high_cardinality], y_train)
    X_test_high_card = high_card_transformer.transform(
        X_test[high_cardinality])

    # Combine all features
    X_train_preprocessed = pd.concat([
        X_train.select_dtypes(include=['int64', 'float64']),
        X_train_low_card,
        X_train_high_card
    ], axis=1)

    X_test_preprocessed = pd.concat([
        X_test.select_dtypes(include=['int64', 'float64']),
        X_test_low_card,
        X_test_high_card
    ], axis=1)

    return X_train_preprocessed, X_test_preprocessed, y_train


def transform_and_scaling_step(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               skew_threshold: float = 0.5,
                               lmbda: float = 0.15) -> Tuple[pd.DataFrame,
                                                             pd.DataFrame,
                                                             pd.Series]:
    """
    Transform and scale the training and test datasets.

    The function identifies the skewed features of the numerical columns of the training dataset and applies a Box-Cox transformation. Then, it scales all numerical features using the StandardScaler. The function transforms both the training and test datasets.

    Parameters:
        X_train (pd.DataFrame): The training dataset.
        X_test (pd.DataFrame): The test dataset.
        y_train (pd.Series): The target variable for the training dataset.
        skew_threshold (float, optional): The threshold for identifying skewed features. Defaults to 0.5.
        lmbda (float, optional): The lambda parameter for the Box-Cox transformation. Defaults to 0.15.

    Returns:
        X_train_preprocessed (pd.DataFrame): The preprocessed training dataset.
        X_test_preprocessed (pd.DataFrame): The preprocessed test dataset.
        y_train (pd.Series): The target variable for the training dataset.
    """
    # Identify numerical columns
    numerical_features = X_train.select_dtypes(
        include=['int64', 'float64']).columns
    skewed_features = identify_skewed_features(
        X_train[numerical_features], threshold=skew_threshold)
    non_skewed_features = [
        col for col in numerical_features if col not in skewed_features]

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

    # Fit on training data and transform both datasets
    X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get feature names after preprocessing
    skewed_feature_names = [clean_feature_name(
        f"{col}_box_cox") for col in skewed_features]
    non_skewed_feature_names = [
        clean_feature_name(col) for col in non_skewed_features]

    feature_names = skewed_feature_names + non_skewed_feature_names

    # Convert to Dataframes
    X_train_preprocessed = pd.DataFrame(
        X_train_preprocessed, columns=feature_names)
    X_test_preprocessed = pd.DataFrame(
        X_test_preprocessed, columns=feature_names)

    return X_train_preprocessed, X_test_preprocessed, y_train


def plot_feature_importance(
        selected_models: dict,
        feature_names: list,
        top_n: int = 20) -> None:
    """
    Plot the top N features by importance for each model in the given dictionary.

    Parameters
    ----------
    selected_models : dict
        A dictionary containing the models to plot, with the model name as the key and the model
        object as the value.
    feature_names : list
        The list of feature names for the models.
    top_n : int, optional
        The number of top features to show in each plot. Defaults to 20.

    Returns
    -------
    None
    """
    _, ax = plt.subplots(1, 2, figsize=(16, 6))

    for i, (model_name, model) in enumerate(selected_models.items()):
        feature_importances = get_feature_importance(model, feature_names)
        top_features = feature_importances.head(top_n)
        top_features.plot.barh(ax=ax[i])
        ax[i].set_title(f"{model_name} top {top_n} features by importance")
        ax[i].invert_yaxis()

    plt.tight_layout()
    plt.show()
