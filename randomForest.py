import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time as t

def write_text_to_pdf(pdf: PdfPages, text_lines: str, lines_per_page=40):
    """
    Prints text line by line to a PDF, starting a new page when necessary.

    Parameters:
        text_lines (list): List of text lines to print.
        output_file (str): The name of the output PDF file.
        lines_per_page (int): Number of lines per page. Default is 40.
    """
    if pdf is not None:
        # Split the text into pages based on the specified lines per page
        for page_num in range(0, len(text_lines), lines_per_page):
            # Create a new figure for each page
            fig, ax = plt.subplots(figsize=(8.5, 11))  # Letter size: 8.5 x 11 inches
            ax.axis('off')  # Turn off the axes

            # Get lines for the current page
            page_lines = text_lines[page_num:page_num + lines_per_page]

            # Add text to the figure
            for i, line in enumerate(page_lines):
                ax.text(0.1, 1 - (i + 1) * 0.025, line, fontsize=10, va='top', ha='left', wrap=True)

            # Save the figure as a page in the PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            


def save_plot_to_pdf(pdf: PdfPages, fig):
    """
    Save the current plot to the PDF.
    """
    pdf.savefig(fig)
    plt.close(fig)


def random_forest(pdf: PdfPages = None):
    """
    Perform Random Forest Regression on the dataset.
    Save analysis and visualizations to a PDF.
    """
    start = t.time()

    try:
        # Load the dataset
        df = pd.read_excel('data/squadData.xlsx')

        # Define predictors and target variable
        X = df[['Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen', 
                'Take-Ons Succ', 'Carries Carries', 'Carries 1/3', 'Receiving Rec']]
        y = df['GF']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=104, shuffle=True
        )

        # Train a Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=104)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error on Test Set: {mse:.2f}")

        # Open the PDF for writing
        if pdf is not None:

            # Add introductory text
            intro_text = (
                "### Random Forest Regression Analysis\n\n"
                f"#### Mean Squared Error on Test Set: {mse:.2f}\n"
            )
            write_text_to_pdf(pdf, intro_text)

            # Plot: Actual vs Predicted values
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='b')
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Reference line
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values (Random Forest)')
            save_plot_to_pdf(pdf, fig)

            # Plot: Feature importances
            feature_importances = model.feature_importances_
            fig, ax = plt.subplots()
            ax.barh(X.columns, feature_importances, color='skyblue')
            ax.set_xlabel('Feature Importance')
            ax.set_title('Feature Importances (Random Forest)')
            save_plot_to_pdf(pdf, fig)

    except Exception as e:
        print(f"An error occurred randomForest.py: {e}")
    finally:
        end = t.time()
        print(f"Time taken: {end - start:.2f} seconds.")


# Run the function to save results to a PDF
