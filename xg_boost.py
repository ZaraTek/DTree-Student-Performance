import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('dataset.xlsx', engine='openpyxl')
print(df.head())
print(df['G3'].value_counts())

df['higher'] = df['higher'].map({'yes': 1, 'no': 0}).fillna(0)
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'internet', 'romantic']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('G3', axis=1)
y = df['G3']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

tree_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=None,
    min_samples_leaf=5,
    min_samples_split=20,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_y_pred = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_y_pred)
print('Decision Tree Accuracy:', tree_accuracy)
print(classification_report(y_test, tree_y_pred))
print(confusion_matrix(y_test, tree_y_pred))

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    num_class=len(set(y_encoded))  )

xgb_model.fit(X_train, y_train)

xgb_y_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
print('XGBoost Accuracy:', xgb_accuracy)
print(classification_report(y_test, xgb_y_pred))
print(confusion_matrix(y_test, xgb_y_pred))

plt.figure(figsize=(8000/100, 3000/100), dpi=100)
tree_plot = plot_tree(tree_model,
                      filled=True,
                      feature_names=X.columns,
                      fontsize=10,
                      proportion=False,
                      precision=2)
plt.savefig('decision_tree.png', format='png', bbox_inches='tight', transparent=True)
plt.show()
