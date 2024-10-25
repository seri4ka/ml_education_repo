
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

# Загрузка набора данных Iris
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Превращаем в задачу бинарной классификации (класс 0 против остальных)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказание значений на тестовой выборке
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Вероятности положительного класса

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные значения')
plt.ylabel('Фактические значения')
plt.show()

# ROC-кривая
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC-кривая (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложные положительные срабатывания')
plt.ylabel('Истинные положительные срабатывания')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
plt.show()

# Визуализация классов по двум признакам
X_2d = X[:, :2]  # Выбор первых двух признаков
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.2, random_state=42)

model_2d = LogisticRegression()
model_2d.fit(X_train_2d, y_train_2d)

# Создание сетки для визуализации границы принятия решений
xx, yy = np.meshgrid(np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 100),
                     np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 100))

# Предсказание на сетке
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Визуализация классов и границы принятия решений
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors='k', cmap='coolwarm', s=100, label='Объекты классов')
plt.title('Логистическая регрессия: Классы и граница принятия решений')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()
plt.show()

# Вывод коэффициентов модели
print(f"Коэффициенты (beta): {model.coef_}")
print(f"Свободный член (beta_0): {model.intercept_}")
print(f"Точность модели: {accuracy:.2f}")
print("Матрица ошибок:")
print(cm)
