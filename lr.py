import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import time as t
from matplotlib.backends.backend_pdf import PdfPages
import traceback
from scipy import stats


def write_text_to_pdf(pdf: PdfPages, text_lines: str, lines_per_page=40, title=False, header=False, fontsize=10, lines_on_page=0, num_pages=0):
    """
    Write text to a PDF file. If the text is too long for one page, it will continue on the next page.
    The function takes the following:
    - pdf: the PDF file to write to
    - text_lines: the text to write
    - lines_per_page: the number of lines per page (default: 40)
    - title: whether the text is a title (default: False)
    - header: whether the text is a header (default: False)
    - fontsize: the font size of the text (default: 10)
    - lines_on_page: the number of lines already written on the current page (default: 0)
    - num_pages: the number of pages already written (default: 0)
    """
    text_lines = text_lines.split('\n')
    if pdf is not None:
        if lines_on_page == 0:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
        else:
            fig = plt.gcf()
            ax = plt.gca()
        
        for line in text_lines:
            if title:
                if lines_on_page > 0 or num_pages > 0:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    num_pages += 1
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')
                ax.text(0.5, 0.5, line, fontsize=16, ha='center', va='center')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                num_pages += 1
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                lines_on_page = 0
                continue

            if lines_on_page >= lines_per_page:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                num_pages += 1
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                lines_on_page = 0

            if header:
                ax.text(0.5, 1 - (lines_on_page + 1) * 0.025, line, fontsize=14, va='top', ha='center')
            else:
                ax.text(0.1, 1 - (lines_on_page + 1) * 0.025, line, fontsize=fontsize, va='top', ha='left')
            lines_on_page += 1

    return [lines_on_page, num_pages]

def increment_counters(arr1: list, arr2: list) -> list:
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            arr1[i] += arr2[i]
    else:
        print("Error: Arrays must be of the same length")
        exit(0)
    return arr1

def save_plot_to_pdf(pdf: PdfPages, fig):
    pdf.savefig(fig)
    plt.close()

def plot_x_vs_y(pdf: PdfPages, X: pd.DataFrame, y: pd.DataFrame):
    for i in range(X.shape[1]):
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, i], y)
        ax.set_xlabel(X.columns[i])
        ax.set_ylabel(y.name)
        ax.set_title(f"{X.columns[i]} vs {y.name}")
        save_plot_to_pdf(pdf, fig)

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i + 1) for i in range(len(X.columns))]
    return vif_data

def feature_reduction_with_pvalues(X: pd.DataFrame, y: pd.DataFrame, vif_threshold=10, pval_threshold=0.05):
    while True:
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        pvalues = model.pvalues[1:]
        vif_data = calculate_vif(X)
        high_vif = vif_data[vif_data["VIF"] > vif_threshold]
        high_pval = pvalues[pvalues > pval_threshold]

        if not high_pval.empty:
            feature_to_remove = high_pval.idxmax()
        elif not high_vif.empty:
            feature_to_remove = high_vif["Variable"].iloc[0]
        else:
            break

        X = X.drop(columns=[feature_to_remove])

    return X
def save_dataframe_to_pdf(pdf: PdfPages, df: pd.DataFrame, title=""):
    """
    Save a pandas DataFrame as a table in a PDF using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('tight')
    ax.axis('off')
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # Create the table at the top
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='top',
                    bbox=[0, 0.2, 1, 0.8])  # [left, bottom, width, height]
    
    # Adjust font size and layout
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    
    # Save to PDF with tight layout
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def linear_regression(pdf: PdfPages, X: pd.DataFrame, y: pd.DataFrame,dataset_name:str,is_y_transformed=False, is_x_transformed=False):
    """
    Perform Multiple Linear Regression on the dataset.
    The function takes the following parameters:
    - pdf: the PDF file to write to
    - X: the features DataFrame
    - y: the target variable DataFrame
    - dataset_name: the name of the dataset
    The function produces a pdf report with the following:
    - Plots of each feature against the target variable
    - Full and reduced model columns
    - Model summary
    
    """
    X_reduced = feature_reduction_with_pvalues(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=104, shuffle=True)
    # Print the reduced columns
    x_reduced_columns = "\n".join([f"{i+1}. {col}" for i, col in enumerate(X_train.columns)])
    # Print the Full Model columns
    x_full_columns = "\n".join([f"{i+1}. {col}" for i, col in enumerate(X.columns)])
    X_train_reduced_with_const = sm.add_constant(X_train)
    final_model = sm.OLS(y_train, X_train_reduced_with_const).fit()
    vif_data_reduced = calculate_vif(X_train)
    X_test_reduced = X_test[X_train.columns]
    X_test_reduced_with_const = sm.add_constant(X_test_reduced)
    y_pred = final_model.predict(X_test_reduced_with_const)
    mse = mean_squared_error(y_test, y_pred)
    # Calculate residuals
    residuals = y_test - y_pred
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_reduced)
    conf_int = final_model.conf_int()
    conf_int.columns = ['Lower CI', 'Upper CI']

    try:
        start = t.time()
        if pdf is not None:
            # Plot the relationship between each feature and the target variable
            totals = [0, 0]
            write_text_to_pdf(pdf, f"Linear Regression for {dataset_name}", title=True, lines_on_page=totals[0], num_pages=totals[1])
            plot_x_vs_y(pdf, X_train, y_train)
            
            if (is_y_transformed) and (is_x_transformed):
                text_to_write = f"Full Final Model (Y, X Transformed):\n{x_full_columns}"
                text_to_write = f"Reduced Final Model (Y, X Transformed):\n{x_reduced_columns}"
            elif (is_y_transformed) and (not is_x_transformed):
                text_to_write = f"Full Final Model (Y Transformed):\n{x_full_columns}"
                text_to_write = f"Reduced Final Model (Y Transformed):\n{x_reduced_columns}"
            elif (not is_y_transformed) and (is_x_transformed):
                text_to_write = f"Full Final Model (X Transformed):\n{x_full_columns}"
                text_to_write = f"Reduced Final Model (X Transformed):\n{x_reduced_columns}"
            else:
                text_to_write = f"Full Final Model\n{x_full_columns}"
                text_to_write = f"Reduced Final Model:\n{x_reduced_columns}"


            text_to_write += f"Model Summary:\n{final_model.summary()}"
            text_to_write += f"VIF for Reduced Model:\n{vif_data_reduced.to_string(index=False)}"
            text_to_write += f"Mean Squared Error on Test Set: {mse:.2f}"
            
            vars = write_text_to_pdf(pdf, text_to_write, lines_on_page=totals[0], num_pages=totals[1])
            totals = increment_counters(totals, vars)

            _, pval, __, _ = het_breuschpagan(residuals, sm.add_constant(y_pred))
            text_to_write = f"Breusch-Pagan test (Equality of variance): p-value: {pval:.3f}\n"
            text_to_write += "Equality of Variance violated (heteroscedasticity detected).\n" if pval < 0.05 else "Equality of Variance holds (homoscedasticity).\n"
            vars = write_text_to_pdf(pdf, text_to_write, lines_on_page=totals[0], num_pages=totals[1])
            totals = increment_counters(totals, vars)
            # ANOVA table
            # anova_table = sm.stats.anova_lm(final_model, typ=2)
            # save_dataframe_to_pdf(pdf, anova_table, title="ANOVA Table")
            text_to_write = f"R^2 Score: {model.score(X_test_reduced, y_test):.2f}"
            # Plot residuals
            write_text_to_pdf(pdf, text_to_write, lines_on_page=totals[0], num_pages=totals[1])
            save_dataframe_to_pdf(pdf, conf_int, title="Confidence Intervals for Betas")
            plt.figure()
            plt.scatter(y_pred, residuals)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            save_plot_to_pdf(pdf, plt.gcf())
            # Perform a QQ-plot
            statistic, p_value = stats.shapiro(residuals)
            plt.figure(figsize=(10, 6))
            sm.qqplot(final_model.resid, line='45')
            plt.title('QQ Plot with 45 deg line\nShapiro-Wilk p-value: {:.4f}'.format(p_value))
            save_plot_to_pdf(pdf, plt.gcf())
            statistic, p_value = stats.shapiro(residuals)
            plt.figure(figsize=(10, 6))
            sm.qqplot(final_model.resid, line='s')
            plt.title('QQ Plot with standardized line\nShapiro-Wilk p-value: {:.4f}'.format(p_value))
            save_plot_to_pdf(pdf, plt.gcf())                   
            # Perform Shapiro-Wilk test on residuals
            # Print results
            pred_actual_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
            save_dataframe_to_pdf(pdf, pred_actual_df, title="Predictions vs Actuals")
            print("Report generated successfully.")
            # Print number of pages
    except Exception as e:
        print("An error occurred in linear_regression:")
        print(f"Error message: {e}")
        print("Traceback details:")
        traceback.print_exc()  # Print the full traceback to the console
    finally:
        end = t.time()
        print(f"Time taken: {end - start:.2f} seconds.")
        plt.close('all')
        return final_model, mse, vif_data_reduced, y_pred, y_test