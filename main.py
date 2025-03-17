import pandas as pd
from lr import linear_regression
from randomForest import random_forest
from decisionTree import decision_tree
from matplotlib.backends.backend_pdf import PdfPages
import tracemalloc
import numpy as np
from sklearn.preprocessing import RobustScaler

def main():
    # Create a PDF file to save plots
    # Snapshot of memory usage
    current, peak = tracemalloc.get_traced_memory()

    df = pd.read_excel('data/squadData.xlsx')

    columns_to_convert = [
        'GF', 'Poss', 'Take-Ons Succ', 'Carries Carries', 'Carries 1/3', 'Receiving Rec',
        'SCA Types PassLive', 'SCA Types PassDead', 'SCA Types Sh',
        'Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen', 'Touches Touches', 'Touches Def 3rd'
    ]
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df_filtered = df.dropna(subset=columns_to_convert)

    X = df_filtered[['Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen', 'Take-Ons Succ', 'Carries Carries', 'Carries 1/3', 'Receiving Rec']]
    y = df_filtered['GF']
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    # Convert X_scaled to a DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    pdf_file = 'output.pdf'
    y_trans_pdf_file = 'output_trans.pdf'
    pdf2= PdfPages(y_trans_pdf_file)
    pdf = PdfPages(pdf_file)

    linear_regression(pdf, X, y, "Soccer squad data, to predict goals")
    pdf.close()

    # Calculate memory usage after running the functions
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory usage after running the functions: {current / 10**6} MB (peak: {peak / 10**6} MB)")
    tracemalloc.stop()
    # Close the PDF file
    pdf.close()

if __name__ == "__main__":
    main()
