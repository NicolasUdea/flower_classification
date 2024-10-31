from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the Iris dataset
'''
The Iris dataset consists of 150 samples of 3 different species of iris
(iris setosa, iris virginica, and iris versicolor)
The features are: petal and sepal length and width
'''
iris = load_iris()

# Convert to DataFrame for easier manipulation and visualization
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Map target values to species names for better visual understanding
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Create a pairplot to visualize the distribution of the data
pairplot = sns.pairplot(df, hue='target', markers=['o', 's', 'D'])

# Save the plot as a high-quality PNG image
pairplot.savefig('pairplot.png', dpi=300)

# Show the plot
plt.show()
