from pandas.io.sql import pandasSQL_builder
from fisher_score import fisher_score
from scipy.stats import zscore
from sklearn.datasets import load_iris
import seaborn as sb
import pandas as pd


data = load_iris()

X = data.data
y = data.target

# normalize data
Xn = zscore(X)

_, idx = fisher_score(Xn,y)
print(f'Fisher Score: {idx}')

print(f'Feature Ranking (higher to lower) {[data.feature_names[i] for i in idx]}')

df = pd.DataFrame(Xn, columns=data.feature_names)
df['target'] = y

g = sb.PairGrid(df, hue="target")
g.map(sb.scatterplot)
g.add_legend()
