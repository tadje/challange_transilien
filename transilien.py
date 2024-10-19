import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib as plt
import csv

data_x = pd.read_csv("train_f_x.csv")
data_y = pd.read_csv("y_train_sncf.csv")
data_x["station"] = pd.factorize(data_x["station"])[0]+1
data_x["date"] = pd.to_datetime(data_x["date"])
data_x["year"] =  data_x["date"].dt.year
data_x["month"] = data_x["date"].dt.month
data_x['day'] = data_x['date'].dt.day
data_x["dayofweek"] = data_x['date'].dt.weekday

X = data_x[['year','day','dayofweek', 'month', 'station', 'vacances','job','ferie']]
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
Y = data_y['y']
# Polynomial_Features = PolynomialFeatures(degree=4)
# poly = Polynomial_Features.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(poly, Y, test_size=0.2, random_state=42)


# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
# print(f"Coefficient de détermination (R²) : {r2:.2f}")
# print(y_pred[10], y_test[10])



data_x = pd.read_csv("train_f_x.csv")
data_y = pd.read_csv("y_train_sncf.csv")
data_x["station"] = pd.factorize(data_x["station"])[0]+1
data_x["date"] = pd.to_datetime(data_x["date"])
data_x["year"] =  data_x["date"].dt.year
data_x["month"] = data_x["date"].dt.month
data_x['day'] = data_x['date'].dt.day
data_x["dayofweek"] = data_x['date'].dt.weekday

scaler = StandardScaler()
X = data_x[['year','day','dayofweek', 'month', 'station', 'vacances','job','ferie']]
Y = data_y['y']
# Polynomial_Features = PolynomialFeatures(degree=4)
# poly = Polynomial_Features.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
# print(f"Coefficient de détermination (R²) : {r2:.2f}")


rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Erreur quadratique moyenne : {mse}")
print(f"Score R² : {r2}")
# Tracé du graphique
# plt.bar(xplot, yplot)
# plt.title("Valeurs pour la station 10 au fil du temps")
# plt.xlabel('Date')
# plt.ylabel('Valeur')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
data_x = pd.read_csv("test_f_x_THurtzP.csv")
data_x["station"] = pd.factorize(data_x["station"])[0]+1
data_x["date"] = pd.to_datetime(data_x["date"])
data_x["year"] =  data_x["date"].dt.year
data_x["month"] = data_x["date"].dt.month
data_x['day'] = data_x['date'].dt.day
data_x["dayofweek"] = data_x['date'].dt.weekday


X = data_x[['year','day','dayofweek', 'month', 'station', 'vacances','job','ferie']]
Y_test_predict = rf.predict(X)
with open('mon_fichier.csv', 'w', newline='') as fichier_csv:
    writer = csv.writer(fichier_csv)
    
    # Écrire les lignes dans le fichier CSV
    writer.writerows([Y_test_predict])


