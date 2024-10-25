
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Генерация синтетических данных для линейной регрессии
X, y, coefficients = make_regression(n_samples=100, n_features=1, noise=10, coef=True, random_state=42)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание значений на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Фактические значения')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Линейная регрессия')
plt.title('Линейная Регрессия')
plt.xlabel('Независимая переменная (X)')
plt.ylabel('Зависимая переменная (y)')
plt.legend()
plt.grid()
plt.show()

# Вывод коэффициентов модели и оценок
print(f"Коэффициент наклона (beta_1): {model.coef_[0]}")
print(f"Свободный член (beta_0): {model.intercept_}")
print(f"Средняя квадратичная ошибка (MSE): {mse}")
print(f"Коэффициент детерминации (R^2): {r2}")
