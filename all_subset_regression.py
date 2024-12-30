import itertools
import statsmodels.api as sm
import pandas as pd
import time
def all_subsets_regression(X: pd.DataFrame, y : pd.DataFrame):
    """
    Perform all subsets regression to evaluate all possible combinations of predictors.
    
    Parameters:
    - X: DataFrame of predictors.
    - y: Series or DataFrame of the target variable.
    
    Returns:
    - results: DataFrame summarizing the metrics for each subset.
    """
    predictors = X.columns
    results = []

    # Iterate through all possible combinations of predictors
    for k in range(1, len(predictors) + 1):
        for subset in itertools.combinations(predictors, k):
            subset = list(subset)
            
            # Fit model with the current subset of predictors
            X_subset = sm.add_constant(X[subset])  # Add intercept
            model = sm.OLS(y, X_subset).fit()
            # Collect performance metrics
            metrics = {
                'Subset': subset,
                'Adjusted R^2': model.rsquared_adj,
                'AIC': model.aic,
                'BIC': model.bic,
                'P-Values': model.pvalues.to_dict()
            }
            results.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by Adjusted R^2 (or other metric)
    results_df = results_df.sort_values(by='Adjusted R^2', ascending=False).reset_index(drop=True)
    
    return results_df

# Example usage
def main():
    # calculate time taken to run the function
    start = time.time()
    # Load the dataset
    df = pd.read_excel('data/squadData.xlsx')

    # Convert columns to numeric
    columns_to_convert = [
        'Poss', 'Touches Touches',
       'Touches Def Pen', 'Touches Def 3rd', 'Touches Mid 3rd',
       'Touches Att 3rd', 'Touches Att Pen','Take-Ons Att',
       'Take-Ons Succ', 'Take-Ons Tkld', 
       'Carries Carries', 'Receiving Rec','GF'
    ]
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    df_filtered = df.dropna(subset=columns_to_convert)

    # Define predictors and target
    X = df_filtered[['Poss', 'Touches Touches',
       'Touches Def Pen', 'Touches Def 3rd', 'Touches Mid 3rd',
       'Touches Att 3rd', 'Touches Att Pen','Take-Ons Att',
       'Take-Ons Succ', 'Take-Ons Tkld', 
       'Carries Carries', 'Receiving Rec']]
    y = df_filtered['GF']

    # Perform all subsets regression
    results_df = all_subsets_regression(X, y)

    # Display the top models
    print("\nTop 5 Models by Adjusted R^2:")
    # print the subset, adjusted R^2, AIC, BIC, and p-values
    print(results_df.head(5))
    
    # display only the subset from the df
    print("\nTop 5 Models by Adjusted R^2:")
    print(results_df['Subset'].head(5))
    # print each subset one by one in the df
    for i in range(5):
        print(results_df['Subset'].iloc[i])
    
    # display only the adjusted R^2 from the df
    print("\nTop 5 Models by Adjusted R^2:")
    print(results_df['Adjusted R^2'].head(5))
    # display only the AIC from the df
    print("\nTop 5 Models by Adjusted R^2:")
    print(results_df['AIC'].head(5))
    # display only the BIC from the df
    print("\nTop 5 Models by Adjusted R^2:")
    print(results_df['BIC'].head(5))
    # display only the p-values from the df
    print("\nTop 5 Models by Adjusted R^2:")
    print(results_df['P-Values'].head(5))
    # calculate time taken to run the function
    end = time.time()
    
    print(f"\nTime taken: {end - start:.2f} seconds.")
    # calculate time in hh:mm:ss
    print(time.strftime("%H:%M:%S", time.gmtime(end - start)))


if __name__ == "__main__":
    main()
