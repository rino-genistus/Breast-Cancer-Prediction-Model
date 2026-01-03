import pandas as pd

data = pd.read_csv('C:/Users/rinog/Downloads/archive/data.csv')
print(data.to_string())
print(f"Data.head(): {data.head()}")
print(f"Data.info(): {data.info()}")
print(f"Data.describe(): {data.describe()}")