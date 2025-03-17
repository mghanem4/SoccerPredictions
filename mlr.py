import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

def feature_reduction_with_pvalues(X: pd.DataFrame, y: pd.DataFrame, vif_threshold=10, pval_threshold=0.05):
    """Perform feature reduction by removing variables with high VIF and high p-values."""
    while True:
        # Add constant for OLS
        X_with_const = sm.add_constant(X)

        # Fit the OLS model and calculate p-values
        model = sm.OLS(y, X_with_const).fit()
        pvalues = model.pvalues[1:]  # Skip the constant

        # Calculate VIF
        vif_data = calculate_vif(X)

        # Find features with high VIF
        high_vif = vif_data[vif_data["VIF"] > vif_threshold]
        high_pval = pvalues[pvalues > pval_threshold]

        # Identify feature to remove: prioritize p-value, then VIF
        if not high_pval.empty:
            feature_to_remove = high_pval.idxmax()  # Remove feature with highest p-value
        elif not high_vif.empty:
            feature_to_remove = high_vif["Variable"].iloc[0]  # Remove first feature with high VIF
        else:
            break  # Exit when no high p-value or VIF remains

        print(f"Removing feature: {feature_to_remove} (p-value: {pvalues.get(feature_to_remove, 'N/A')}, "
              f"VIF: {vif_data[vif_data['Variable'] == feature_to_remove]['VIF'].values[0]})")

        # Drop the selected feature
        X = X.drop(columns=[feature_to_remove])

    return X

def mlr():
    # Clear terminal before running (Windows)
    start = t.time()

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

    # Perform feature reduction
    X_train_reduced = feature_reduction_with_pvalues(X_train, y_train)

    # Train a final model
    X_train_reduced_with_const = sm.add_constant(X_train_reduced)
    final_model = sm.OLS(y_train, X_train_reduced_with_const).fit()

    # Print summary of the final model
    print(final_model.summary())

    # Calculate VIF for the reduced model
    page_break("VIF for reduced model:")
    vif_data_reduced = calculate_vif(X_train_reduced)
    print(vif_data_reduced)

    # Evaluate the model on the test set
    X_test_reduced = X_test[X_train_reduced.columns]
    X_test_reduced_with_const = sm.add_constant(X_test_reduced)
    y_pred = final_model.predict(X_test_reduced_with_const)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse:.2f}")
    residuals = y_test - y_pred

    page_break("Breusch-Pagan test:\n")
    _, pval, __, f_pval = het_breuschpagan(residuals, sm.add_constant(y_pred))
    print(f"Breusch-Pagan p-value: {pval:.3f}")
    if pval < 0.05:
        print("Equality of Variance violated (heteroscedasticity detected).")
    else:
        print("Equality of Variance holds (homoscedasticity).")

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train_reduced, y_train)
    y_pred = model.predict(X_test_reduced)
    print(f"Mean Squared Error on Test Set: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R^2 Score: {model.score(X_test_reduced, y_test):.2f}")
    page_break("")
    # Print predicted values vs actual values to terminal
    print("Predicted Values vs Actual Values")
    print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
    end = t.time()
    # Get the confidence intervals for each coefficient
    conf_int = final_model.conf_int()
    conf_int.columns = ['Lower CI', 'Upper CI']
    print(conf_int)
    print(f"Time taken: {end - start:.2f} seconds.")
    # Caclulate time taken to run the script in hh:mm:ss
    print(t.strftime("%H:%M:%S", t.gmtime(end - start)))
    # Perform QQ plot and show the plot
    plt.figure(figsize=(10, 6))
    sm.qqplot(final_model.resid, line='s')
    plt.title('Normal Q-Q Plot')
    plt.show()
    return final_model, mse, vif_data_reduced, y_pred, y_test


if __name__ == "__main__":
    mlr()