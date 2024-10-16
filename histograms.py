import pandas as pd
import matplotlib.pyplot as plt
import os

# load dataset
df_x = pd.read_excel('dataset.xlsx', engine='openpyxl')
df_x.to_csv('dataset.csv', index=False)
df = pd.read_csv('dataset.csv')

histograms_dir = 'Histograms'
if not os.path.exists(histograms_dir):
    os.makedirs(histograms_dir)

# create histograms
def plot_histograms(df):
    for col in df.columns:
        plt.figure()
        if df[col].dtype in ['int64', 'float64']:
            df[col].hist(bins=15)
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel('Frequency')
        else:
            df[col].value_counts().plot(kind='bar')
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel('Count')
        plt.savefig(f'{histograms_dir}/{col}_histogram.png')
        plt.show()
plot_histograms(df)
