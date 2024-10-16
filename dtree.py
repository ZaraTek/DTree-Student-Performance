import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris  # Example dataset

# load dataset
df = pd.read_excel('dataset.xlsx', engine='openpyxl')
print(df.head())
print(df['G3'].value_counts())

# preprocess data
df['higher'] = df['higher'].map({'yes': 1, 'no': 0}).fillna(0)
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'internet', 'romantic']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# select features, set g3 as target
X = df.drop('G3', axis=1)  # Adjust if different features or target are desired
y = df['G3']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# initilize dtree regressors
tree_model = DecisionTreeRegressor(random_state=42)

# train model
tree_model.fit(X_train, y_train)

# feature importance
importances = tree_model.feature_importances_
print(importances)

# predict and evaluate
y_pred = tree_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_importance = sorted(zip(tree_model.feature_importances_, X.columns), reverse=True)
print(feature_importance[:10])  # top 10 features
print("All features:", sum(importances))
print("Top 10 features:", sum([imp for imp, _ in feature_importance[:10]]))

# visualize and save dtree
plt.figure(figsize=(15000/100, 3000/100), dpi=100)
tree_plot = tree.plot_tree(tree_model,
                           filled=True,
                           feature_names=X.columns,
                           fontsize=10,
                           proportion=False,
                           precision=2)
plt.savefig('decision_tree_wide.png', format='png', bbox_inches='tight')  
plt.show()
