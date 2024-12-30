import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import os
import time as t
def page_break(string: str):
    print("\n" * 10)
    print(string)

def calculate_vif(X: pd.DataFrame)->pd.DataFrame:
    """Calculate Variance Inflation Factor for each feature."""
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i + 1) for i in range(len(X.columns))]
    return vif_data

def random_forrest():
    start = t.time()
    # Clear terminal before running (Windows)

    # Load the data
    df = pd.read_excel('data/squadData.xlsx')

    # Convert specific columns to numeric
    columns_to_convert = [
        'GF', 'Poss', 'Take-Ons Succ', 'Carries Carries', 'Carries 1/3', 'Receiving Rec',
        'SCA Types PassLive', 'SCA Types PassDead', 'SCA Types Sh',
        'Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen', 'Touches Touches', 'Touches Def 3rd'
    ]
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    df_filtered = df.dropna(subset=columns_to_convert)

    # Define predictors and target variable
    X = df_filtered[['Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen', 'Take-Ons Succ', 'Carries Carries', 'Carries 1/3', 'Receiving Rec']]
    y = df_filtered['GF']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=104, shuffle=True)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=104)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print("Random Forest Predicted Values vs Actual Values")
    print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

    # Feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importance)
    # visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()
    end = t.time()
    print(f"Time taken: {end - start:.2f} seconds.")
    # Caclulate time taken to run the script in hh:mm:ss
    print(t.strftime("%H:%M:%S", t.gmtime(end - start)))
    return rf_model, mse, feature_importance, y_pred, y_test

