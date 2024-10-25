
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Генерация синтетических данных для регрессии
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Разделение данных на обучающую и тестовую выборки
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Создание и обучение модели дерева решений для регрессии
model_reg = DecisionTreeRegressor()
model_reg.fit(X_train_reg, y_train_reg)

# Предсказание значений на тестовой выборке
y_pred_reg = model_reg.predict(X_test_reg)

# Оценка модели
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

# Визуализация результатов
plt.figure(figsize=(10, 6))

# Отображение фактических значений
plt.scatter(X_test_reg, y_test_reg, color='blue', marker='o', label='Фактические значения')

# Отображение предсказанных значений
plt.plot(X_test_reg, y_pred_reg, color='red', marker='x', label='Предсказанные значения', linestyle='None')

# Построение линии регрессии для наглядности
X_line = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
y_line = model_reg.predict(X_line)
plt.plot(X_line, y_line, color='green', label='Линия регрессии')

plt.title('Дерево решений для регрессии')
plt.xlabel('Независимая переменная (X)')
plt.ylabel('Зависимая переменная (y)')
plt.legend()
plt.grid()
plt.show()

print(f"Средняя квадратичная ошибка (MSE): {mse:.2f}")
print(f"Коэффициент детерминации (R^2): {r2:.2f}")
