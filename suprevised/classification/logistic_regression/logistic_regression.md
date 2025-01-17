## Шпаргалка по логистической регрессии

### 1. Основы логистической регрессии
Логистическая регрессия — это статистический метод, используемый для бинарной классификации, который моделирует вероятность принадлежности наблюдения к определенному классу. Она основана на использовании логистической функции для преобразования линейной комбинации независимых переменных в вероятность.

### 2. Математическая модель
Формула логистической регрессии для двух классов выглядит следующим образом:

\[
P(Y=1|X) = \sigma(Z) = \frac{1}{1 + e^{-Z}}
\]

где:
- \(Y\) — зависимая переменная (класс),
- \(X\) — независимые переменные,
- \(Z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n\) — линейная комбинация независимых переменных,
- \(\sigma(Z)\) — логистическая функция (сигмоида).

### 3. Интерпретация коэффициентов
Коэффициенты \(\beta\) можно интерпретировать как изменение логарифма шансов (odds) на 1 единицу изменения независимой переменной:

\[
\text{odds} = \frac{P(Y=1|X)}{P(Y=0|X)} = e^Z
\]

### 4. Оценка параметров модели
Параметры модели оцениваются с использованием метода максимального правдоподобия. Задача состоит в том, чтобы максимизировать функцию правдоподобия:

\[
L(\beta) = \prod_{i=1}^{m} P(Y_i|X_i) = \prod_{i=1}^{m} \sigma(Z_i)^{Y_i} (1 - \sigma(Z_i))^{(1 - Y_i)}
\]

где \(m\) — количество наблюдений.

### 5. Оценка качества модели
- **Коэффициент детерминации (Pseudo-R²)**: Различные варианты (например, McFadden R²) для оценки качества модели:
  
\[
R^2 = 1 - \frac{L(\beta)}{L(0)}
\]

где \(L(0)\) — функция правдоподобия для модели с постоянным членом.

- **Матрица ошибок**: Позволяет визуализировать количество правильно и неправильно классифицированных наблюдений.

- **ROC-кривая**: Используется для оценки качества классификации, показывающая зависимость между истинными положительными и ложными положительными значениями.

### 6. Допущения логистической регрессии
1. **Линейная связь**: Линейная связь между независимыми переменными и логарифмом шансов.
2. **Независимость**: Наблюдения должны быть независимыми.
3. **Отсутствие мультиколлинеарности**: Независимые переменные не должны быть сильно коррелированы между собой.

### 7. Визуализация
- **График зависимости**: Визуализация вероятностей принадлежности к классу с помощью логистической функции и фактических данных.

### 8. Пример использования
На практике логистическая регрессия может быть использована для:
- Определения вероятности того, что клиент совершит покупку (да/нет).
- Оценки вероятности заболевания на основе различных медицинских показателей.

### Заключение
Логистическая регрессия является мощным инструментом для бинарной классификации, который просто интерпретируется и используется в различных областях, включая медицину, маркетинг и финансовый анализ.
