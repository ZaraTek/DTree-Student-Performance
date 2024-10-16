import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_excel('dataset.xlsx', engine='openpyxl')
print(df.head())
print(df['G3'].value_counts())

df['higher'] = df['higher'].map({'yes': 1, 'no': 0}).fillna(0)
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'internet', 'romantic']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('G3', axis=1)
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=None,
    min_samples_leaf=5,
    min_samples_split=20,
    random_state=42
)

tree_model.fit(X_train, y_train)

importances = tree_model.feature_importances_
print(importances)

y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

feature_importance = sorted(zip(tree_model.feature_importances_, X.columns), reverse=True)
print("Top 10 features:", feature_importance[:10])
print("All features:", sum(importances))
print("Top 10 features' importance sum:", sum([imp for imp, _ in feature_importance[:10]]))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None, 
    min_samples_split=2,  
    min_samples_leaf=1,  
    max_features='sqrt', 
    random_state=42  
)

rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print('Random Forest Accuracy:', rf_accuracy)
print(classification_report(y_test, rf_y_pred))
print(confusion_matrix(y_test, rf_y_pred))

rf_importances = rf_model.feature_importances_
sorted_rf_importance = sorted(zip(rf_importances, X.columns), reverse=True)
print("Random Forest Top 10 features:", sorted_rf_importance[:10])

plt.figure(figsize=(8000/100, 3000/100), dpi=100)
tree_plot = tree.plot_tree(tree_model,
                           filled=True,
                           feature_names=X.columns,
                           fontsize=10,
                           proportion=False,
                           precision=2)
plt.savefig('random_forest_decision_tree.png', format='png', bbox_inches='tight', transparent = True)
plt.show()
