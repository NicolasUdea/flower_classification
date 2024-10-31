from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargamos el dataset
'''
El conjunto de datos Iris consta de 150 muestras de 3 especies diferentes de iris 
(iris setosa, iris virginica e iris versicolor)
Las características son: longitud y ancho de petalos y sépalos
'''
iris = load_iris()

# Convertir a DataFrame para manipulación y visualización
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

# Añadimso nombres de clases para mejorar comprensión visual
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Grafiacamos las características de las especies de iris
sns.pairplot(df, hue='target', markers=['o', 's', 'D'])
plt.show()

# Grafica de caja
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f'{feature} vs target')
plt.tight_layout()
plt.show()
