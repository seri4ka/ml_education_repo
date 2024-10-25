
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Загрузка набора данных Iris
iris = load_iris()
X = iris.data[:, :2]  # Используем только первые два признака для визуализации
y = iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели дерева решений
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Предсказание значений на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Матрица ошибок для классификации')
plt.xlabel('Предсказанные значения')
plt.ylabel('Фактические значения')
plt.show()

# Визуализация классов на 2D графике
plt.figure(figsize=(10, 6))

# График с цветами классов
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=100, label='Объекты классов')
plt.title('Классификация с использованием деревьев решений')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

# Построение границы принятия решений
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# Добавление легенды
plt.legend(*scatter.legend_elements(), title='Классы')
plt.grid()
plt.show()

print(f"Точность модели: {accuracy:.2f}")
