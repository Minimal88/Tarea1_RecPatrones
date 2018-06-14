import pandas as pd

url = "C:/Users/gollo/gitRepos/Tarea1_RecPatrones/data/MPI_national.csv"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['ISO','Country','MPI Urban','Headcount Ratio Urban','Intensity of Deprivation Urban','MPI Rural','Headcount Ratio Rural','Intensity of Deprivation Rural'])



from sklearn.preprocessing import StandardScaler
features = ['MPI Urban','Headcount Ratio Urban','Intensity of Deprivation Urban','MPI Rural','Headcount Ratio Rural','Intensity of Deprivation Rural']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['ISO']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
	, columns = ['principal component 1', 'principal component 2'])



finalDf = pd.concat([principalDf, df[['MPI Urban']]], axis = 1)



import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()



pca.explained_variance_ratio_