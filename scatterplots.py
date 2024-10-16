import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load dataset
df_x = pd.read_excel('dataset.xlsx', engine='openpyxl')
df_x.to_csv('dataset.csv', index=False)
df = pd.read_csv('dataset.csv')

selected_columns = df[['absences', 'studytime', 'higher', 'freetime', 'G3']]
selected_columns['higher'] = selected_columns['higher'].map({'yes': 1, 'no': 0})

sns.pairplot(selected_columns)
plt.suptitle('Scatter Matrix of Selected Variables', y=1.02)  # Adjust title and its position
plt.savefig(fname = 'Blue Scatter Matrix.png', transparent = True)
plt.show()
