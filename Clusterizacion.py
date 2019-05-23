import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

datos = pd.read_csv('Wholesale_customers_data.csv')
print datos.head()
categ_features = ['Channel', 'Region']
cont_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']


os.getcwd()
#obtener los datos segun estadistica descriptiva
#mayor cantidad se gasta en promedio en Fresh Groceries
print datos[cont_features].describe()

#one hot encoding
for col in categ_features:
    #se concatenaran los datos en col
    dummies = pd.get_dummies(datos[col], prefix=col)
    #concatena los datos en columnas
    datos = pd.concat([datos, dummies], axis=1)
    #reemplaza los datos de la columna por el nuevo encoding
    datos.drop(col, axis=1, inplace=True)
print datos.head()

#normalización de datos
mms = MinMaxScaler()
mms.fit(datos)
datos_norm = mms.transform(datos)
print datos_norm

#within clusters sum of squares
sum_squared_dist = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(datos_norm)
    sum_squared_dist.append(km.inertia_)

plt.plot(K, sum_squared_dist, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method')
plt.show()

#valor de k=5 aproximadamente

kmeans = KMeans(n_clusters=5).fit(datos_norm)
centroids = kmeans.cluster_centers_
print(centroids)

# Predicting the clusters
labels = kmeans.predict(datos)
print labels

