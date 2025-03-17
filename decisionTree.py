import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import time as t

def write_text_to_pdf(pdf: PdfPages, text_lines: str, lines_per_page=40):
    """
    Prints text line by line to a PDF, starting a new page when necessary.

    Parameters:
        pdf (PdfPages): The PdfPages object to save the PDF to.
        text_lines (str): Text to print, with lines separated by '\n'.
        lines_per_page (int): Number of lines per page. Default is 40.
    """
    text_lines = text_lines.split('\n')
    
    if pdf is not None:
        # Initialize variables for page handling
        lines_on_page = 0
        page_num = 0
        # Loop through the text lines
        for i, line in enumerate(text_lines):
            # Create a new page if necessary
            if lines_on_page == 0 or lines_on_page == lines_per_page:
                if lines_on_page > 0:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)  # Close previous figure to free memory
                # Create a new figure for the next page
                fig, ax = plt.subplots(figsize=(8.5, 11))  # Letter size: 8.5 x 11 inches
                ax.axis('off')  # Turn off the axes
                lines_on_page = 0  # Reset lines counter
                page_num += 1

            # Calculate the vertical position for the text on the current page
            ax.text(0.1, 1 - (lines_on_page + 1) * 0.025, line, fontsize=10, va='top', ha='left', wrap=True)
            lines_on_page += 1

        # Save the last page if there are any remaining lines
        if lines_on_page > 0:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def save_plot_to_pdf(pdf, fig):
    pdf.savefig(fig)
    plt.close()

def decision_tree(pdf: PdfPages = None):
    """
    Perform Decision Tree Regression on the dataset.
    This function saves the analysis and visualizations to a PDF.
    """
    start = t.time()
    df = pd.read_excel('data/squadData.xlsx')

    # Define predictors and target variable
    X = df[['Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen', 'Take-Ons Succ', 'Carries Carries', 'Carries 1/3', 'Receiving Rec']]
    y = df['GF']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=104, shuffle=True)

    # Train a Decision Tree model
    model = DecisionTreeRegressor(random_state=104)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse:.2f}")

    if pdf is not None:
        # Add introductory text
        intro_text = "Decision Tree Regression Analysis\n\nModel Summary:\n"
        intro_text += f"Mean Squared Error on Test Set: {mse:.2f}\n\n"

        # Save the model performance plot to the PDF
        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='b')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values (Decision Tree)')
        save_plot_to_pdf(pdf, plt.gcf())  # Save current figure

        # Additional plot: Visualize the decision tree
        plt.figure(figsize=(12, 8))
        plot_tree(model, filled=True, feature_names=X.columns, rounded=True)
        plt.title('Decision Tree Visualization')
        save_plot_to_pdf(pdf, plt.gcf())  # Save current figure
        write_text_to_pdf(pdf, intro_text)
    end = t.time()
    print(f"Time taken: {end - start:.2f} seconds.")
