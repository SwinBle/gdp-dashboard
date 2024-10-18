import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="DataInsightPro - Advanced Analytics Dashboard", layout="wide")

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def data_overview(df):
    st.header("1. Data Overview")
    st.write(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns")
    st.write("First few rows of the dataset:")
    st.write(df.head())

    st.subheader("1.1 Column Information")
    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.write(col_info)

def data_quality_check(df):
    st.header("2. Data Quality Assessment")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_table = pd.concat([missing, missing_pct], axis=1, keys=['Missing Values', 'Percentage'])
    missing_table = missing_table[missing_table['Missing Values'] > 0].sort_values('Percentage', ascending=False)

    if not missing_table.empty:
        st.subheader("2.1 Missing Values")
        st.write(missing_table)
        fig = px.bar(missing_table, x=missing_table.index, y='Percentage', title='Percentage of Missing Values by Column')
        st.plotly_chart(fig)
    else:
        st.write("No missing values found in the dataset.")

    # Duplicates
    duplicates = df.duplicated().sum()
    st.subheader("2.2 Duplicate Rows")
    st.write(f"Number of duplicate rows: {duplicates}")

def perform_eda(df):
    st.header("3. Exploratory Data Analysis (EDA)")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    st.subheader("3.1 Numeric Variables Distribution")
    selected_numeric = st.multiselect("Select numeric columns for distribution analysis:", numeric_cols)
    for col in selected_numeric:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig)

    st.subheader("3.2 Correlation Analysis")
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, title="Correlation Heatmap")
    st.plotly_chart(fig)

    st.subheader("3.3 Categorical Variables Analysis")
    selected_categorical = st.multiselect("Select categorical columns for analysis:", categorical_cols)
    for col in selected_categorical:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = ['Category', 'Count']
        fig = px.bar(value_counts, x='Category', y='Count', title=f"Distribution of {col}")
        st.plotly_chart(fig)

    return corr

def feature_engineering(df):
    st.header("4. Feature Engineering")

    # Categorical encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoding_method = st.selectbox("Select encoding method for categorical variables:",
                                   ["Label Encoding", "One-Hot Encoding"])
    for col in categorical_cols:
        if encoding_method == "Label Encoding":
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
            st.write(f"Applied Label Encoding to '{col}'")
        else:
            one_hot = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, one_hot], axis=1)
            df.drop(col, axis=1, inplace=True)
            st.write(f"Applied One-Hot Encoding to '{col}'")

    # Numeric scaling
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaling_method = st.selectbox("Select scaling method for numeric variables:",
                                  ["StandardScaler", "MinMaxScaler"])
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    st.write(f"Applied {scaling_method} to numeric columns")

    st.write("Updated dataframe preview:")
    st.write(df.head())

    return df

def prepare_data_for_modeling(df, target_column):
    st.subheader("Data Preparation for Modeling")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check for non-numeric columns in features
    non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
    if len(non_numeric_cols) > 0:
        st.warning(f"The following columns contain non-numeric data: {', '.join(non_numeric_cols)}")
        handling_method = st.radio("How would you like to handle non-numeric columns?",
                                   ("Drop", "Encode", "Cancel modeling"))

        if handling_method == "Drop":
            X = X.select_dtypes(include=['int64', 'float64'])
            st.info(f"Dropped non-numeric columns: {', '.join(non_numeric_cols)}")
        elif handling_method == "Encode":
            for col in non_numeric_cols:
                X[col] = pd.Categorical(X[col]).codes
            st.info(f"Encoded non-numeric columns: {', '.join(non_numeric_cols)}")
        else:
            st.stop()

    # Check target column
    if not pd.api.types.is_numeric_dtype(y):
        st.error(f"The target column '{target_column}' is not numeric. Please select a numeric target for regression.")
        return None, None

    st.success("Data prepared successfully for modeling.")
    return X, y

def train_models(X, y):
    st.header("5. Model Training and Evaluation")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "ElasticNet": ElasticNet(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42),
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "Support Vector Machine": SVR(),
        "Neural Network (MLP)": MLPRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(random_state=42)
    }

    results = {}

    for name, model in models.items():
        st.subheader(f"5.{list(models.keys()).index(name) + 1} {name}")

        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        results[name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "Explained Variance Score": evs,
            "CV R2": cv_scores.mean()
        }

        st.write(f"Model Performance:")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"R2 Score: {r2:.4f}")
        st.write(f"Explained Variance Score: {evs:.4f}")
        st.write(f"Cross-Validation R2 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Residual plot
        residuals = y_test - y_pred
        fig = px.scatter(x=y_test, y=residuals, title=f"Residual Plot - {name}")
        st.plotly_chart(fig)

    st.write("Model training complete.")

    return results

def generate_insights(df, results):
    st.header("6. Generate Insights")

    st.subheader("6.1 Data Insights")

    # Basic statistics for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.write("Basic Statistics for Numeric Columns:")
        st.write(numeric_df.describe())

        # Correlation insights for numeric columns
        corr = numeric_df.corr()
        high_corr = corr[abs(corr) > 0.7].stack().reset_index()
        high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
        if not high_corr.empty:
            st.write("High Correlations between Numeric Columns:")
            st.write(high_corr)
        else:
            st.write("No high correlations found between numeric columns.")
    else:
        st.write("No numeric columns found in the dataset.")

    # Categorical column insights
    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        st.write("Categorical Column Insights:")
        for col in categorical_cols:
            st.write(f"Unique values in {col}: {df[col].nunique()}")
            st.write(df[col].value_counts().head())

    # Model performance insights
    st.subheader("6.2 Model Performance Insights")
    if results:
        best_model = max(results, key=lambda x: results[x]['R2'])
        st.write(f"The best performing model is {best_model} with an R2 score of {results[best_model]['R2']:.4f}")

        # Visualize model comparison
        model_comparison = pd.DataFrame(results).T
        fig = px.bar(model_comparison, y=model_comparison.index, x='R2', title="Model R2 Score Comparison")
        st.plotly_chart(fig)

        # Generate recommendations
        st.subheader("6.3 Recommendations")
        recommendations = [
            "Consider feature selection to focus on the most important variables.",
            f"Investigate why {best_model} performed best and consider ensemble methods.",
            "Collect more data if possible to improve model performance.",
            "Explore non-linear relationships in the data.",
            "Consider trying more advanced models or deep learning approaches."
        ]
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.write("No model results available. Please run the model training step.")

def main():
    st.title("DataInsightPro - Advanced Analytics Dashboard")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            data_overview(df)
            data_quality_check(df)
            perform_eda(df)
            df_engineered = feature_engineering(df)

            if st.checkbox("Proceed with Model Training?"):
                target_column = st.selectbox("Select target column for modeling", df.columns)
                X, y = prepare_data_for_modeling(df_engineered, target_column)

                if X is not None and y is not None:
                    try:
                        model_results = train_models(X, y)
                        generate_insights(df, model_results)
                    except Exception as e:
                        st.error(f"An error occurred during model training: {str(e)}")
                        st.write("Please check your data and ensure all columns used for modeling contain valid numeric values.")
                else:
                    st.error("Failed to prepare data for modeling. Please check your data and try again.")
            else:
                generate_insights(df, None)

if __name__ == "__main__":
    main()
