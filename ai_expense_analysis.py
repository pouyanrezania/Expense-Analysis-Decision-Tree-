
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv', parse_dates=['Date'])
df['YearMonth'] = df['Date'].dt.to_period('M').dt.to_timestamp()
monthly = df.groupby(['YearMonth','Category'])['Amount'].sum().reset_index()
monthly['MonthIdx'] = (monthly['YearMonth'].dt.year - monthly['YearMonth'].dt.year.min())*12 + monthly['YearMonth'].dt.month
pivot = monthly.pivot(index='MonthIdx', columns='Category', values='Amount').fillna(0).sort_index()
pivot.index.name = 'MonthIdx'

def make_supervised(data, n_lags=2):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i].flatten())
        y.append(data[i])
    return np.array(X), np.array(y)

data_matrix = pivot.values
n_lags = 2
X, y = make_supervised(data_matrix, n_lags=n_lags)
split = int(len(X)*0.8) if len(X)>1 else 0
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
model = DecisionTreeRegressor(random_state=42)

if len(X_train) == 0:
    model.fit(X, y)
    pred = model.predict(X[-1].reshape(1, -1))
    print("Not enough history for train/test split. Predicted next month amounts:")
    for cat, val in zip(pivot.columns, pred.flatten()):
        print(f"  {cat}: {val:.2f}")
else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) if len(y_test)>0 else float('nan')
    mae = mean_absolute_error(y_test, y_pred) if len(y_test)>0 else float('nan')
    print(f"R2 score on test set: {r2:.3f}")
    print(f"MAE on test set: {mae:.2f}")
    next_input = data_matrix[-n_lags:].flatten().reshape(1, -1)
    next_pred = model.predict(next_input).flatten()
    print("\nPredicted next month spending by category:")
    for cat, val in zip(pivot.columns, next_pred):
        print(f"  {cat}: {val:.2f}")

top3 = pivot.sum().sort_values(ascending=False).head(3).index.tolist()
for cat in top3:
    plt.figure()
    plt.plot(pivot.index, pivot[cat].values, marker='o', label='Historical')
    try:
        plt.plot([pivot.index.max()+1], [next_pred[list(pivot.columns).index(cat)]], marker='x', label='Predicted')
    except:
        pass
    plt.title(f"Spending history & prediction - {cat}")
    plt.xlabel('MonthIdx')
    plt.ylabel('Amount')
    plt.legend()
    plt.show()
