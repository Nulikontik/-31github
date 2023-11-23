Загрузка…
CodeSquare
Перейти к основному контенту
Копия блокнота "lab_1_ml.ipynb"
Копия блокнота "lab_1_ml.ipynb"_Пометка блокнота снята
Последнее сохранение: 18:47
[ ]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
Intro to Pandas
[ ]
col_1 = [1, 2, 34, 5, 56, 7, 8, 45]
col_2 = [100, 101, 102, 103, 104, 105, 106, 107]
col_3 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
df = pd.DataFrame({'col_1': col_1, 'col_2': col_2, 'col_3': col_3})
df
account_circle

[ ]
df
account_circle

[ ]
df.col_1 + df.col_2
account_circle
0    101
1    103
2    136
3    108
4    160
5    112
6    114
7    152
dtype: int64
[ ]
df.col_1.median()
account_circle
7.5
[ ]
(170+160+2000+140)/4
account_circle
617.5
[ ]

[ ]

Part 1
Regression
У нас история развития бизнеса относительно доходности.

[ ]
year = [2016, 2017, 2018, 2019, 2020]
profit = [2000, 2500, 3100, 3600, 4000]
df = pd.DataFrame({'year': year, 'profit': profit})
df
account_circle

[ ]
sns.scatterplot(x=df.year, y=df.profit)
account_circle

[ ]
sns.lineplot(x=df.year, y=df.profit)
account_circle

Давайте предскажем доход бизнеса в 2023 году, если будем считать данный тренд развития неизменным.

[ ]
degree = 1
coeff = np.polyfit(df.year, df.profit, degree)
coeff
account_circle
array([ 5.10000e+02, -1.02614e+06])
Мы получили два числа, на базе которых мы можем написать линейную зависимость дохода от года в виде уравнения (формулы)

𝑝𝑟𝑜𝑓𝑖𝑡=510∗𝑦𝑒𝑎𝑟−1.03∗106

А вот и сами коэффициенты

[ ]
print('Slope: ', coeff[0])
print('intersept: ', coeff[1])
account_circle
Slope:  509.9999999999752
intersept:  -1026139.9999999494
Теперь в полученную формулу подставим год и получим результат:

[ ]
profit_2023 = coeff[0]*2040 + coeff[1]
[ ]
print(f'В 2040 году доходность составит {round(profit_2023)} $')
account_circle
В 2040 году доходность составит 14260 $
[ ]

Part 2
Clastering
[ ]
arr = np.random.randn(60, 2)
arr[np.random.choice(arr.shape[0], size=30), :] += 5
[ ]
df = pd.DataFrame({'X': arr[:, 0], 'Y': arr[:, 1]})
df['class'] = np.nan
Мы сгенерили набор чисел и оформили его в виде пандас таблицы.

[ ]
df.head(10)
account_circle

[ ]
sns.scatterplot(x= df.X, y = df.Y)
account_circle

[ ]
sns.kdeplot(df.X)
sns.kdeplot(df.Y)
account_circle

[ ]
sns.scatterplot(x= df.X, y = arr[:, 1])
plt.axvline(x=df.X.min(), color='g', linestyle='--')
plt.axvline(x=df.X.max(), color='r', linestyle='--')
plt.axhline(y=df.Y.min(), color='g', linestyle='--')
plt.axhline(y=df.Y.max(), color='r', linestyle='--')
account_circle

Граница - это прямая, которая пройдет по диагонали

[ ]
sns.scatterplot(x= df.X, y = arr[:, 1])
plt.axvline(x=df.X.min(), color='g', linestyle='--')
plt.axvline(x=df.X.max(), color='r', linestyle='--')
plt.axhline(y=df.Y.min(), color='g', linestyle='--')
plt.axhline(y=df.Y.max(), color='r', linestyle='--')
sns.lineplot(x = [df.X.min(), df.X.max()], y = [df.Y.max(), df.Y.min()], color='m')
account_circle

[ ]
x = [df.X.min(), df.X.max()]
y = [df.Y.max(), df.Y.min()]
degree = 1

coeff = np.polyfit(x, y, degree)
coeff
account_circle
array([-1.14211529,  5.38494002])
Уравнение прямой имеет вид:

𝑦=𝑐𝑜𝑒𝑓𝑓[0]𝑥+𝑐𝑜𝑒𝑓𝑓[1]

Тогда кластер, который находится выше прямой будет соответствовать условию:

𝑦>𝑐𝑜𝑒𝑓𝑓[0]𝑥+𝑐𝑜𝑒𝑓𝑓[1]

[ ]
df['class'] = np.nan
df.loc[df.Y > coeff[0]*df.X + coeff[1], 'class'] = 'A'
df.loc[df.Y < coeff[0]*df.X + coeff[1], 'class'] = 'B'
df.head(8)
account_circle

[ ]
sns.scatterplot(x='X', y='Y', data=df, hue='class')
account_circle

Part 3
Self worf
Давайте попробуем разбить на виды цветок рода Ирис.

[ ]
df = sns.load_dataset('iris')
[ ]
df = df.loc[df.species!= 'virginica', ['sepal_length', 'sepal_width']]
[ ]
df['species']=np.nan
[ ]
df
account_circle

[ ]
sns.scatterplot(x=df.sepal_length, y = df.sepal_width)
sns.lineplot(x=[4.5, 6], y=[2.5, 3.7])
account_circle

[ ]
# Your code
degree = 1
coeff = np.polyfit([4.5, 6], [2.5, 3.7], degree)
coeff
account_circle
array([ 0.8, -1.1])
[ ]
df.loc[df.sepal_width > coeff[0]*df.sepal_length + coeff[1], 'species'] = 'A'
df.loc[df.sepal_width < coeff[0]*df.sepal_length + coeff[1], 'species'] = 'B'
[ ]
sns.scatterplot(x=df.sepal_length, y = df.sepal_width, hue=df.species)
account_circle

[ ]
df
account_circle

[ ]

K-means
Это уже готовый алгоритм кластеризации.

[ ]
from sklearn.cluster import KMeans
[ ]
kmeans = KMeans(n_clusters=2)
kmeans.fit(df[['sepal_length', 'sepal_width']])
account_circle

[ ]
labels = kmeans.labels_
[ ]
labels
account_circle
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=int32)
[ ]
df['k_means_clast'] = labels
[ ]
df
account_circle

[ ]
sns.scatterplot(x=df.sepal_length, y = df.sepal_width, hue=df.k_means_clast)
account_circle

[ ]

Платные продукты Colab - Отменить подписку
