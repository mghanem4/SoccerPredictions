# Perform a linear relationship test for GF scored and Poss
#
# # Load the data inside the data/squadData.xlsx directory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# clear terminal before running (Windows)
os.system('cls')

# Load the data
df = pd.read_excel('data/squadData.xlsx')

print(df.keys())

# Convert both GF and Poss to numeric, setting non-numeric values to NaN
df['GF'] = pd.to_numeric(df['GF'], errors='coerce')
df['Poss'] = pd.to_numeric(df['Poss'], errors='coerce')
df['Take-Ons Succ'] = pd.to_numeric(df['Take-Ons Succ'], errors='coerce')
df['Carries Carries'] = pd.to_numeric(df['Carries Carries'], errors='coerce')
df['Carries 1/3'] = pd.to_numeric(df['Carries 1/3'], errors='coerce')
df['Receiving Rec'] = pd.to_numeric(df['Receiving Rec'], errors='coerce')
df['SCA Types PassLive'] = pd.to_numeric(df['SCA Types PassLive'], errors='coerce')
df['SCA Types PassDead'] = pd.to_numeric(df['SCA Types PassDead'], errors='coerce')
df['SCA Types Sh'] = pd.to_numeric(df['SCA Types Sh'], errors='coerce')
df['Touches Mid 3rd'] = pd.to_numeric(df['Touches Mid 3rd'], errors='coerce')
df['Touches Att Pen'] = pd.to_numeric(df['Touches Att Pen'], errors='coerce')
df['Touches Att 3rd'] = pd.to_numeric(df['Touches Att 3rd'], errors='coerce')


# Drop rows with missing values in the GF and Poss columns
df_filtered = df.dropna(subset=['GF','Take-Ons Succ','Carries Carries','Carries 1/3','Receiving Rec','SCA Types PassLive','SCA Types PassDead','SCA Types Sh','Touches Mid 3rd',
       'Touches Att 3rd', 'Touches Att Pen'])
# Perform a linear regression test
X = df_filtered[['Touches Mid 3rd','Touches Att 3rd', 'Touches Att Pen','Take-Ons Succ','Carries Carries','Carries 1/3','Receiving Rec']]

y = df_filtered['GF']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=104, shuffle=True)

# Create a linear regression model
model = LinearRegression()
# Fit the model
model.fit(X, y)
# Make predictions
y_pred = model.predict(X)
# Calculate the mean squared error
mse = mean_squared_error(y, y_pred)
# Calculate the R2 score
r2 = r2_score(y, y_pred)

# Print the mean squared error and R2 score as a new page
print("\n\n\n\n\n\n\n\nMSE & R², adj-R²\n")

print('Mean squared error: ', mse)
print('R² score: ', r2)

n = len(X_test)  # Number of observations
p = X_train.shape[1]  # Number of predictors

# Calculate Adjusted R²
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"Adjusted R²: {adj_r2:.2f}")

print(X.keys())


# Plot each X against y (each feature against the target)
for feature in X.keys():
    plt.figure(figsize=(10,6))
    plt.scatter(X[feature], y, label=feature)
    plt.xlabel(feature)
    plt.ylabel('GF')
    plt.title(f'{feature} vs GF')
    plt.legend()
    plt.show()

# Perform a equality of variance test

# Calculate residuals
residuals = y - y_pred

# Scatter plot of residuals vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Poss')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

# Breusch-Pagan test
print("\n\n\n\n\n\n\n\nBreusch-Pagan test:\n")
_, pval, __, f_pval = het_breuschpagan(residuals, sm.add_constant(y_pred))
print(f"Breusch-Pagan p-value: {pval:.3f}")

if pval < 0.05:
    print("Equality of Variance violated (heteroscedasticity detected).")
else:
    print("Equality of Variance holds (homoscedasticity).")


# QQ plot
plt.figure(figsize=(8, 6))
stats.probplot(residuals.values, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()



# Add a constant to the predictors
X_train_with_const = sm.add_constant(X_train)

# Fit the OLS model
ols_model = sm.OLS(y_train, X_train_with_const).fit()

# Get the summary, including p-values for each predictor
print(ols_model.summary())

# print page breaks to view VIF
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nVIF: \n")

# Calculate VIF for each predictor
X_with_const = sm.add_constant(X_train)
vif = pd.DataFrame()
vif["Variable"] = X_train.columns
vif["VIF"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(len(X_train.columns))]

print(vif)
print("\n\n")
# Check each VIF if it is greater than 10
for i in range(len(vif)):
    if vif['VIF'][i] > 10:
        print(f"High VIF detected for {vif['Variable'][i]}")
    else:
        print(f"VIF for {vif['Variable'][i]} < 10")

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDropping the following:\nReceiving Rec,Carries 1/3, Take-Ons Succ, Carries Carries :\n")

# Filter the df to exclude the columns with high VIF
df_reduced = df_filtered.drop(['Receiving Rec', 'Carries 1/3', 'Take-Ons Succ', 'Carries Carries'], axis=1)

# Drop Touches Att 3rd
X_train_reduced = X_train.drop(columns=['Receiving Rec','Carries 1/3', 'Take-Ons Succ','Carries Carries'])

# Add constant for OLS
X_train_reduced_with_const = sm.add_constant(X_train_reduced)


X2 = df_reduced[['Touches Mid 3rd','Touches Att 3rd', 'Touches Att Pen']]

y2 = df_reduced['GF']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=104, shuffle=True)

model2 = LinearRegression()
# Fit the model
model2.fit(X2, y2)
# Make predictions
y_pred2 = model2.predict(X2)


# Make page break to print y_pred2
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n y_pred2: \n")
print(y_pred2)

# Refit OLS model
ols_model_reduced = sm.OLS(y2_train, X_train_reduced_with_const).fit()

# Print summary
print(ols_model_reduced.summary())


print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nVIF Reduced: \n")

# Calculate VIF for reduced model
vif_reduced = pd.DataFrame()
vif_reduced["Variable"] = X_train_reduced.columns
vif_reduced["VIF"] = [variance_inflation_factor(X_train_reduced_with_const.values, i+1) for i in range(len(X_train_reduced.columns))]
print(vif_reduced)


# Drop Touches Att 3rd
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDropping the following:\nTouches Att 3rd:\n")





# Filter the df to exclude the columns with high VIF
df_reduced2 = df_filtered.drop(['Touches Att 3rd'], axis=1)


X3 = df_reduced2[['Touches Mid 3rd','Touches Att Pen']]

y3 = df_reduced2['GF']

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=104, shuffle=True)

model3 = LinearRegression()
# Fit the model
model3.fit(X3, y3)
# Make predictions
y_pred3 = model3.predict(X3)

# Make page break to print y_pred3
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n y_pred3: \n")
print(y_pred3)

X_train_reduced_2 = X_train_reduced.drop(columns=['Touches Att 3rd'])
# Add constant for OLS
X_train_reduced_2_with_const = sm.add_constant(X_train_reduced_2)
# Refit OLS model
ols_model_reduced_2 = sm.OLS(y_train, X_train_reduced_2_with_const).fit()
# Print summary
print(ols_model_reduced_2.summary())

# Calculate VIF for reduced model
vif_reduced_2 = pd.DataFrame()
vif_reduced_2["Variable"] = X_train_reduced_2.columns
vif_reduced_2["VIF"] = [variance_inflation_factor(X_train_reduced_2_with_const.values, i+1) for i in range(len(X_train_reduced_2.columns))]
print(vif_reduced_2)


