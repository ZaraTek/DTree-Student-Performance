import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# load dataset
df_x = pd.read_excel('dataset.xlsx', engine='openpyxl')
df_x.to_csv('dataset.csv', index=False)
df = pd.read_csv('dataset.csv')

df.fillna(0, inplace=True)

# split data into features and target
X = df.drop('G3', axis=1)
y = df['G3']

X = pd.get_dummies(X, drop_first=True)
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

importances = model.feature_importances_
features = X.columns

indices = np.argsort(importances)[::-1]

# plotting
plt.figure(figsize=(10, 8))
plt.title('Random Forest Feature Importance')
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
plt.show()
plt.savefig(fname = 'Random Forest Feature Importance.png', transparent = True)
