import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree

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
df_encoded = pd.concat([df_filtered], axis=1)

league_dummies = pd.get_dummies(df_filtered['League'], prefix='League', drop_first=True)
df_encoded = pd.concat([df_encoded], axis=1)


# Define x and y
features = ['SCA SCA90'] 
X = df_encoded[features]
y = df_encoded[ 'GCA GCA90']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tree = DecisionTreeRegressor(max_depth=5, random_state=42)

# Train the model
tree.fit(X_train, y_train)

# Predict on test data
y_pred = tree.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

n = len(X_test)  # Number of observations
p = X_train.shape[1]  # Number of predictors

# Calculate Adjusted R²
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"Adjusted R²: {adj_r2:.2f}")

print(f"fetures: {features}")