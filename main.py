import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

apple = pd.read_csv('Apple.csv')
apple.dropna(inplace=True)
y = apple['Close']
x = apple.drop(columns=['Adj Close', 'Date', 'Close', 'Volume'])
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

def calculate_fitness():
    global X_train, y_train, X_test
    regr = MLPRegressor(max_iter=200)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test) #Fazendo a predição no conjunto de teste

    return mean_squared_error(y_test, y_pred)