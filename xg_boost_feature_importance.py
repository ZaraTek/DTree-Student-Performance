import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_excel('dataset.xlsx', engine='openpyxl')

df['higher'] = df['higher'].map({'yes': 1, 'no': 0}).fillna(0)
categorical_cols = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'internet', 'romantic'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('G3', axis=1)
y = df['G3']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['G3'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

importance = model.get_booster().get_score(importance_type='weight')

sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
features, values = zip(*sorted_importance)

colors = plt.cm.Purples(np.linspace(0.5, 1, len(values)))

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(features))
bars = ax.barh(y_pos, values, color=colors, align='center', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.invert_yaxis() 
ax.set_xlabel('F Score')
ax.set_title('XGBoost Feature Importance')
plt.tight_layout()

plt.savefig('feature_importance_corrected_borders.png', format='png', dpi=300, bbox_inches='tight', transparent = True)
plt.show()
