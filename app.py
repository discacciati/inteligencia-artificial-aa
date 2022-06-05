from sklearn import datasets  #import datasets from sklearn
import pandas as pd  # package for data processing, CSV file I/O library
import numpy as np   # package for linear algebra
import warnings      # a current version of seaborn generates a bunch of warnings that we'll ignore 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns  #package with a Python graphing library
from sklearn.tree import DecisionTreeClassifier  #package to assemble the classification tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

plt.style.use('ggplot')
#Above are the imports of the packages and libraries that will be used


#Iris(plant) is a genus of 260–300 species of flowering plants with showy flowers. 
# It takes its name from the Greek word for a rainbow, which is also the name for the 
# Greek goddess of the rainbow, Iris.

# Sepals Protective parts of the flower, external to the petals, often green, 
# that form the calyx. They are foliate structures, normally smaller and more consistent 
# than the petals, and in most cases they have the primary function of protecting the floral 
# bud, closing over it before anthesis.



#We'll load the Iris flower dataset
iris = datasets.load_iris() #the iris dataset is loaded

dir(datasets)

#Visualizando dados de iris
iris

#Tipos de dados 
type(iris)

#Tipo de variáveis
type("a")


#print iris data keys
#print(iris.keys())

#print the iris data target_name
#print(iris['target_names'])
#print(type(iris.target_names))
#print the iris Description
iris.DESCR

#verificando a quantidade de features_names
iris["feature_names"]

# type(iris["features_names"])


type(iris.data),type(iris.target)

#Verificando a quantidade de linhas e colunas do dataset iris
iris.data.shape

#Verificando o conteudo do dataset. Estes dados serão utilizados para treinar o algoritimo
iris.data

#Dados que serão utilizados para previsão no futuro
iris.target

#Preparando os dados para treinamento 
X = iris.data
y = iris.target
z = iris.target_names[0]
#print(z)

#Criando um dataframe com pandas para iniciar o treinamento dos dados
df = pd.DataFrame(X,columns=iris.feature_names)

#visualizando as primeiras linhas do dataset
df.head()

#Montando uma matriz para visualizar a correlação entre as variaveis
pd.plotting.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='D')




#Iniciando o processo de classificação de acordo com as caracteristicas de cada flor


#Classificador Knn
graph = sns.pairplot(df)

graph

df

knn = KNeighborsClassifier(n_neighbors=6)

type(knn)
# print(type(knn))
knn.fit(iris['data'],iris['target'])

knn.fit(iris['data'],iris['target'])

iris['target']

iris['data']

iris['data'].shape

iris['target'].shape

#print('Prediction {}'.format(prediction))

# iris['target_names'][prediction]

iris['target']

knn.score(iris['data'],iris['target'])
# print(knn.score(iris['data'],iris['target']))
# Filters

# type: Setosa


#3 listas

filter_setosa = [] 
filter_versicolor = [] 
filter_virginica = [] 

#
for item in iris['target']:
     
    if item == 0:
        filter_setosa.append(item)
    if item == 1:
        filter_versicolor.append(item)
    if item == 2:
        filter_virginica.append(item)
    

print(filter_setosa)
print(filter_versicolor)
print(filter_virginica)

#pip install -r requirements.txt   funciona igual o yarn install

#print(list(filterSetosa))

# type: Versicolor
filterVersicolor = filter(lambda iris: iris.target_names == 'versicolor', iris)

# print(list(filterVersicolor))

# type: virginica
filterVirginica = filter(lambda iris: iris.target_names == 'virginica', iris)

# print(list(filterVirginica))

#Para executar o programa, digitar no console: 
#python app.py
