import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel('dataset.xlsx', engine='openpyxl')

X = df[['G2', 'absences', 'age']]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

stratified_cv = StratifiedKFold(n_splits=5)

classifier = DecisionTreeClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 10, 20, 40, 80],
    'min_samples_leaf': [1, 2, 5, 10, 20]
}

grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=stratified_cv, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best training score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test set accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
