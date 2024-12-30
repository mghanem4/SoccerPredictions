import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures

# load data
df = pd.read_excel('gca.xlsx')

# Convert both Age and SCA SCA90 to numeric, setting non-numeric values to NaN
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['SCA SCA90'] = pd.to_numeric(df['SCA SCA90'], errors='coerce')
# Adding in SCA Types PassLive to the model, convert it to numeric and set non-numeric values to NaN
df['SCA Types PassLive'] = pd.to_numeric(df['SCA Types PassLive'], errors='coerce')
# Adding in 90s to the model, convert it to numeric and set non-numeric values to NaN
df['90s'] = pd.to_numeric(df['90s'], errors='coerce')
# Adding in SCA Types PassDead to the model, convert it to numeric and set non-numeric values to NaN
df['SCA Types PassDead'] = pd.to_numeric(df['SCA Types PassDead'], errors='coerce')
# Adding in 'GCA GCA90' to the model, convert it to numeric and set non-numeric values to NaN
df['GCA GCA90'] = pd.to_numeric(df['GCA GCA90'], errors='coerce')
# Adding in 'SCA Types Fld' to the model, convert it to numeric and set non-numeric values to NaN
df['SCA Types Fld'] = pd.to_numeric(df['SCA Types Fld'], errors='coerce')
# Adding in SCA Types Def to the model, convert it to numeric and set non-numeric values to NaN
df['SCA Types Def'] = pd.to_numeric(df['SCA Types Def'], errors='coerce')
# Adding in SCA SCA to the model, convert it to numeric and set non-numeric values to NaN
df['SCA SCA'] = pd.to_numeric(df['SCA SCA'], errors='coerce')
# Adding in SCA Types TO to the model, convert it to numeric and set non-numeric values to NaN
df['SCA Types TO'] = pd.to_numeric(df['SCA Types TO'], errors='coerce')
# Adding in GCA Types PassLive to the model, convert it to numeric and set non-numeric values to NaN
df['GCA Types PassLive'] = pd.to_numeric(df['GCA Types PassLive'], errors='coerce')
# Adding in GCA Types TO to the model, convert it to numeric and set non-numeric values to NaN
df['GCA Types TO'] = pd.to_numeric(df['GCA Types TO'], errors='coerce')
# Adding in GCA Types Sh to the model, convert it to numeric and set non-numeric values to NaN
df['GCA Types Sh'] = pd.to_numeric(df['GCA Types Sh'], errors='coerce')
# Adding in GCA Types Fld to the model, convert it to numeric and set non-numeric values to NaN
df['GCA Types Fld'] = pd.to_numeric(df['GCA Types Fld'], errors='coerce')
# Adding in GCA Types Def to the model, convert it to numeric and set non-numeric values to NaN
df['GCA Types Def'] = pd.to_numeric(df['GCA Types Def'], errors='coerce')

upper_limit = df['SCA SCA90'].quantile(0.99)


# Filter the data to exclude extreme outliers
df_filtered = df[df['SCA SCA90'] <= upper_limit]

# Check the new max value
print(df_filtered['SCA SCA90'].max())
print("\n\n Columns: \n")
print(df_filtered.keys())



# Drop rows with missing values in the Age, SCA SCA90, Position, and 90s columns
df_filtered = df_filtered.dropna(subset=['Age', 'SCA SCA90', 'Position', '90s', 'SCA Types PassLive','SCA Types PassDead', 'GCA GCA90','SCA Types Fld', 'League','SCA Types Def','SCA SCA','SCA Types TO','GCA Types PassLive','GCA Types TO','GCA Types Sh','GCA Types Fld','GCA Types Def'])

print("\n\n \n")
# print df_filtered df number of rows.
print(df_filtered.shape[0])

print("\n\n \n")


# Create dummies for the Position column and league column
position_dummies = pd.get_dummies(df_filtered['Position'], prefix='Position', drop_first=True)
df_encoded = pd.concat([df_filtered, position_dummies], axis=1)

league_dummies = pd.get_dummies(df_filtered['League'], prefix='League', drop_first=True)
df_encoded = pd.concat([df_encoded, league_dummies], axis=1)


# Define features and target
X = df_encoded[['Age', '90s', 'SCA SCA90', 'SCA Types PassLive', 'SCA Types PassDead', 'SCA Types TO', 'SCA Types Sh', 'SCA Types Fld', 'SCA Types Def']]
y = df_encoded['GCA GCA90']
y_log = np.log1p(y)  # log(1 + y) to handle potential 0 values safely

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test data
y_pred = lr.predict(X_test)

# Calculate residuals (using the transformed y)
residuals = y_test - y_pred

# Perform Breusch-Pagan test
_, pval, __, f_pval = het_breuschpagan(residuals, sm.add_constant(y_pred))
print(f"Breusch-Pagan p-value: {pval:.3f}")

# Equality of Variance
if pval < 0.05:
    print("Equality of Variance violated (heteroscedasticity detected).")
else:
    print("Equality of Variance holds.")

# Scatter plot of residuals vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (Log Transformed y)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# QQ plot of residuals
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals (Log Transformed y)')
plt.show()
