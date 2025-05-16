import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def plot_x_vs_y(X: pd.DataFrame, y: pd.DataFrame):
    pdf = PdfPages('plot_x_vs_y.pdf')
    for i in range(X.shape[1]):
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, i], y)
        ax.set_xlabel(X.columns[i])
        ax.set_ylabel(y.name)
        ax.set_title(f"{X.columns[i]} vs {y.name}")
        pdf.savefig(fig)
    pdf.close()
def calculate_vif(X_with_const: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i + 1) for i in range(len(X.columns))]
    return vif_data

def feature_reduction_with_pvalues(X: pd.DataFrame, y: pd.DataFrame, vif_threshold=10, pval_threshold=0.05):
    """
    Feature reduction algorithm, that removes values based on pvalues and VIF
    - X: x column to reduce
    - y: y column to reduce
    - vif_threshold=10:  VIF threshold is set to 10 by default
    - pval_threshold=0.05: p-value threshold is set to 0.05 by default
    """
    while True:
# Add constant B_0
        X_with_const = sm.add_constant(X)
# fit the model
        model = sm.OLS(y, X_with_const).fit()
# fetch p-values
        pvalues = model.pvalues[1:]
# calculate the VIF for each X
        vif_data = calculate_vif(X_with_const, X)
# Store High VIF values
        high_vif = vif_data[vif_data["VIF"] > vif_threshold]
# Store High p-values
        high_pval = pvalues[pvalues > pval_threshold]
# exclude high pvalues & VIF values
        if not high_pval.empty:
            feature_to_remove = high_pval.idxmax()
        elif not high_vif.empty:
            feature_to_remove = high_vif["Variable"].iloc[0]
        else:
            # Once there is no high values, break out. Either return empty list or reduced X
            break
        X = X.drop(columns=[feature_to_remove])
    return X

def linear_regression(X: pd.DataFrame, y: pd.DataFrame, dataset_name: str):
    # Reduce the model
    X_reduced = feature_reduction_with_pvalues(X, y)
    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=104, shuffle=True
    )
    # Fit train model using scikit-learn LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Predict on train and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate MSE for both
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    print(f"Train MSE: {mse_train}")
    print(f"Test MSE: {mse_test}")
    # Calculate residuals
    residuals = y_test - y_pred_test
    # Get coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_
    print(f"Intercept: {intercept}")
    print(f"Coefficients: {coef}")

    return model, mse_train,mse_test, residuals
