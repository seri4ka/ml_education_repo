
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Генерация синтетических данных для множественной линейной регрессии
X_multi, y_multi, coefficients_multi = make_regression(n_samples=100, n_features=2, noise=10, coef=True, random_state=42)

# Разделение данных на обучающую и тестовую выборки
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Создание и обучение модели множественной линейной регрессии
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

# Предсказание значений на тестовой выборке
y_pred_multi = model_multi.predict(X_test_multi)

# Оценка модели
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

# Визуализация результатов в 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Создание сетки для визуализации плоскости регрессии
x1_range = np.linspace(X_multi[:, 0].min(), X_multi[:, 0].max(), 10)
x2_range = np.linspace(X_multi[:, 1].min(), X_multi[:, 1].max(), 10)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
Y_grid = model_multi.intercept_ + model_multi.coef_[0] * X1_grid + model_multi.coef_[1] * X2_grid

# Плоскость регрессии
ax.plot_surface(X1_grid, X2_grid, Y_grid, alpha=0.5, rstride=100, cstride=100, color='red')

# Точки на графике
ax.scatter(X_test_multi[:, 0], X_test_multi[:, 1], y_test_multi, color='blue', label='Фактические значения')
ax.scatter(X_test_multi[:, 0], X_test_multi[:, 1], y_pred_multi, color='green', label='Предсказанные значения')

ax.set_title('Множественная линейная Регрессия')
ax.set_xlabel('Независимая переменная 1 (X1)')
ax.set_ylabel('Независимая переменная 2 (X2)')
ax.set_zlabel('Зависимая переменная (y)')
ax.legend()
plt.show()

# Вывод коэффициентов модели и оценок
print(f"Коэффициенты (beta): {model_multi.coef_}")
print(f"Свободный член (beta_0): {model_multi.intercept_}")
print(f"Средняя квадратичная ошибка (MSE): {mse_multi}")
print(f"Коэффициент детерминации (R^2): {r2_multi}")
