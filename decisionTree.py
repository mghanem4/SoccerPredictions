import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import os
import time as t

def page_break(string: str):
    print("\n" * 10)
    print(string)

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train and evaluate a Decision Tree Regressor, then return the model and its performance."""
    # Train a Decision Tree Regressor model
    dt_model = DecisionTreeRegressor(random_state=104)
    dt_model.fit(X_train, y_train)

    # Predict on test set
    y_pred_dt = dt_model.predict(X_test)

    # Calculate Mean Squared Error for Decision Tree
    mse_dt = mean_squared_error(y_test, y_pred_dt)

    # Feature importance from Decision Tree
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': dt_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return dt_model, mse_dt, feature_importance, y_pred_dt, y_test

def decision_tree():
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

    # Train the Decision Tree model and return it along with performance metrics
    dt_model, mse_dt, feature_importance, y_pred_dt, y_test = train_decision_tree(X_train, y_train, X_test, y_test)

    # Print model performance metrics
    print(f"Decision Tree Mean Squared Error on Test Set: {mse_dt:.2f}")
    print("Decision Tree Predicted Values vs Actual Values")
    print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_dt}))

    # Print Feature Importance
    page_break("Feature Importance from Decision Tree:")
    print(feature_importance)
    end = t.time()
    # Optional: Visualize the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance (Decision Tree)')
    plt.show()
    print(f"Time taken: {end - start:.2f} seconds.")
    # Calculate time taken to run the script in hh:mm:ss
    print(t.strftime("%H:%M:%S", t.gmtime(end - start)))

    # Return the trained model and performance metrics
    return dt_model, mse_dt, feature_importance