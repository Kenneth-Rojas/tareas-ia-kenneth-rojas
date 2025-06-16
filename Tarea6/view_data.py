import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Advertising.csv")
print(data.head())
sns.pairplot(data)
plt.show()
