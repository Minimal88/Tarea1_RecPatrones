import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


url = "C:/Users/gollo/gitRepos/Tarea1_RecPatrones/data/MPI_national_labeled.csv"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['ISO','Country','MPI Urban','Headcount Ratio Urban','Intensity of Deprivation Urban','MPI Rural','Headcount Ratio Rural','Intensity of Deprivation Rural','Nivel'])


#print(df)

features = ['MPI Urban','Headcount Ratio Urban','Intensity of Deprivation Urban','MPI Rural','Headcount Ratio Rural','Intensity of Deprivation Rural']
# Separating out the features
x = df.loc[:, features].values


# Separating out the target
y = df.loc[:,['Nivel']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

#print(y)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
	, columns = ['principal component 1', 'principal component 2'])



finalDf = pd.concat([principalDf, df[['Nivel']]], axis = 1)




fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PCA 1', fontsize = 15)
ax.set_ylabel('PCA 2', fontsize = 15)
ax.set_title('2 componenentes de PCA', fontsize = 20)
countries = ['Nivel_1', 'Nivel_2', 'Nivel_3']
colors = ['r', 'g', 'b']
for country, color in zip(countries,colors):
    indicesToKeep = finalDf['Nivel'] == country
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(countries)
ax.grid()




plt.show()


pca.explained_variance_ratio_