import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Your data
data = {
    'Area (m2)': [6.00, 500.00, 400.00],
    'Land Use': ['Industrial', 'Residential (IZHS)', 'Residential (IZHS)'],
    'Location': ['Asan-Chek, S.U. Kurmanzhan Datka', 'Osh city, Kene-Sai district', 'Osh city, T. Satylganova street'],
    'Price (KGS)': [40000, 1500000, 4000000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Explore the data
print(df)

# Scatter plot: Price vs Area
sns.scatterplot(x='Area (m2)', y='Price (KGS)', data=df, hue='Land Use')
plt.title('Price vs Area')
plt.show()

# Boxplot: Price vs Location
plt.figure(figsize=(12, 6))
sns.boxplot(x='Location', y='Price (KGS)', data=df)
plt.title('Price vs Location')
plt.xticks(rotation=45, ha='right')
plt.show()

# Correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Predictive model
# Convert categorical features to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Land Use', 'Location'], drop_first=True)

# Select features (X) and target variable (y)
X = df_encoded[['Area (m2)', 'Land Use_Residential (IZHS)', 'Location_Osh city, Kene-Sai district', 'Location_Osh city, T. Satylganova street']]
y = df_encoded['Price (KGS)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Example prediction
new_data = {
    'Area (m2)': [300.00],
    'Land Use_Residential (IZHS)': [1],
    'Location_Osh city, Kene-Sai district': [0],
    'Location_Osh city, T. Satylganova street': [0]
}

new_df = pd.DataFrame(new_data)
new_pred = model.predict(new_df)

print(f'Predicted Price for the new data: {new_pred[0]} KGS')
'''
"Площадь, м2: 6,00; Земля: Промназначения; С/у курманжан датка, село Асан-Чек",40 KGS,
"Земля: Сельхозназначения (СНТ, ДНТ); Ош шаарга караштуу, Кен-Сай массивинде.Ориентир:Жапалак",no price,
"Земля: Поселения (ИЖС); Ул. Суркеева, село Курманжан Датка, Асанчек айыл. Ошская область",no price,
"Площадь, м2: 500,00; Земля: Поселения (ИЖС); Ош ш, кенеш",,
Земля: Поселения (ИЖС);,,
Земля: Поселения (ИЖС); г.ош 225квартал жил массив Керме ТОО,,
"Земля: Поселения (ИЖС); г.Ош, жил массив Керме-Тоо, 225-й квартал.",no price,
"Площадь, м2: 400,00; Земля: Поселения (ИЖС); г.Ош, ул.Т.Сатылганова",,
"Площадь, м2: 500,00; Земля: Поселения (ИЖС); город Ош, Фуркат",,
"Площадь, м2: 6 000,00; Земля: Промназначения; Ош шаары Массив Кең сай",no price,
"Площадь, м2: 6,00; Земля: Промназначения; Массив Кең сай Ош",,
"Площадь, м2: 6,00; Земля: Промназначения; Массив Кең сай",,
"Площадь, м2: 6,00; Земля: Промназначения; Кең сай",,
Земля: Поселения (ИЖС); Гульбар,,
"Площадь, м2: 6,00; Земля: Промназначения; Жапалак",,
Земля: Поселения (ИЖС);,,
"Площадь, м2: 20,00; Земля: Поселения (ИЖС); мкр Эдельвейс",,
Земля: Поселения (ИЖС); Супаналиева,no price,
Земля: Поселения (ИЖС);,,
Земля: Поселения (ИЖС);,no price,
"Площадь, м2: 60,00; Земля: Сельхозназначения (СНТ, ДНТ); Фуркат, Кыргыз чек айылы.",no price,
Земля: Поселения (ИЖС);,,
Земля: Поселения (ИЖС); ,,
Земля: Поселения (ИЖС);,,
'''
