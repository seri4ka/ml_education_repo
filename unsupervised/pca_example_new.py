
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Загрузка набора данных Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Функция для визуализации перехода из высокоразмерного пространства в низкоразмерное
def plot_high_to_low_dimension(X, y, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5, edgecolors='k')
    plt.title('Исходное пространство (4D) - проекция на 2 признака')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, edgecolors='k')
    plt.title(f'PCA - {n_components} главных компонент')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Визуализация перехода из 4D в 2D (первые 2 компоненты)
plot_high_to_low_dimension(X_scaled, y, n_components=2)

# Визуализация потери информации при различных количествах главных компонент
def plot_explained_variance(X):
    pca = PCA()
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
    plt.title('Объясненная дисперсия при различном количестве компонент')
    plt.xlabel('Количество главных компонент')
    plt.ylabel('Кумулятивная объясненная дисперсия')
    plt.grid()
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% объясненной дисперсии')
    plt.axvline(x=np.argmax(cumulative_variance >= 0.95) + 1, color='g', linestyle='--',
                label='Минимум компонент для 95%')
    plt.legend()
    plt.show()

# Визуализация объясненной дисперсии
plot_explained_variance(X_scaled)
