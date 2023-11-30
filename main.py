import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def regression_model_evaluation(data, description_column='Description', price_column='Price', degree=2):
    # Преобразование текстовых описаний в числовые признаки
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(data[description_column])

    # Разделение данных на признаки (X) и целевую переменную (y)
    X_numeric = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())

    # Преобразование столбца 'Price' в числовой формат
    data[price_column] = data[price_column].replace({'KGS': ''}, regex=True).str.replace(' ', '').astype(float)

    X = pd.concat([X_numeric, data[[price_column]]], axis=1)

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X.drop(price_column, axis=1), X[price_column], test_size=0.2, random_state=42)

    # Линейная регрессия
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    # Полиномиальная регрессия
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    # Градиентный бустинг
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    # Оценка моделей
    models = {
        'Linear Regression': y_pred_linear,
        f'Polynomial Regression (Degree {degree})': y_pred_poly,
        'Gradient Boosting': y_pred_gb
    }

    results = {}

    for model_name, y_pred in models.items():
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = (abs((y_test - y_pred) / y_test)).mean() * 100

        results[model_name] = {
            'RMSE': rmse,
            'R-squared': r2,
            'MAE': mae,
            'MAPE': mape
        }

        # Визуализация результатов
        plt.scatter(y_test, y_pred, color='black')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linewidth=3)
        plt.xlabel('True Prices')
        plt.ylabel('Predicted Prices')
        plt.title(f'{model_name} - Price Prediction')
        plt.show()

    return results

# Пример использования
file_path = '/datas2.xlsx'
data = pd.read_excel(file_path)
evaluation_results = regression_model_evaluation(data)
print(evaluation_results)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def regression_model_evaluation(data, description_column='Description', price_column='Price', degree=2):
    # Преобразование текстовых описаний в числовые признаки
    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(data[description_column])

    # Разделение данных на признаки (X) и целевую переменную (y)
    X_numeric = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())

    # Преобразование столбца 'Price' в числовой формат
    data[price_column] = data[price_column].replace({'KGS': ''}, regex=True).str.replace(' ', '').astype(float)

    X = pd.concat([X_numeric, data[[price_column]]], axis=1)

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X.drop(price_column, axis=1), X[price_column], test_size=0.2, random_state=42)

    # Линейная регрессия
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    # Полиномиальная регрессия
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    # Градиентный бустинг
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    # Оценка моделей
    models = {
        'Linear Regression': y_pred_linear,
        f'Polynomial Regression (Degree {degree})': y_pred_poly,
        'Gradient Boosting': y_pred_gb
    }

    results = {}

    for model_name, y_pred in models.items():
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = (abs((y_test - y_pred) / y_test)).mean() * 100

        results[model_name] = {
            'RMSE': rmse,
            'R-squared': r2,
            'MAE': mae,
            'MAPE': mape
        }

        # Визуализация результатов
        plt.scatter(y_test, y_pred, color='black')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linewidth=3)
        plt.xlabel('True Prices')
        plt.ylabel('Predicted Prices')
        plt.title(f'{model_name} - Price Prediction')
        plt.show()

    return results

# Пример использования
file_path = '/datas2.xlsx'
data = pd.read_excel(file_path)
evaluation_results = regression_model_evaluation(data)
print(evaluation_results)
fig = plt.figure(figsize=(17, 15))
grid = GridSpec(ncols=1, nrows=2, figure=fig)

ax1 = fig.add_subplot(grid[0, :])
sns.countplot(x=data.date.dt.month, ax=ax1)

ax2 = fig.add_subplot(grid[1, :])
sns.boxplot(x=data.date.dt.month, y='price', data=data, ax=ax2)