import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('tourism_data_500_points.csv')
df.head()

df.info()
df.describe()

x = df.drop('Number_of_Visitors', axis=1)
y = df['Number_of_Visitors']

x_train, y_train, x_test, y_test = train_test_split(x, y, random_state=20)

model = Lasso(alpha=0.1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

r2 = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('MSE :', r2)
print('MAE :', mae)

open with('results.txt', 'w') as f:
    f.write('MSE :', r2 /n 'MAE :', mae /n)
