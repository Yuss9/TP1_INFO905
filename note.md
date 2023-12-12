# Rapport sur le TP : Sélection de Modèle de Prédiction pour le Prix des Maisons

## Introduction

Le but de ce TP est de créer des modèles de prédiction pour le prix des maisons en utilisant trois approches différentes : la régression linéaire, l'arbre de décision, et les forêts aléatoires. Nous allons explorer plusieurs étapes essentielles du processus, de la préparation des données à la recherche d'un modèle optimal à l'aide de GridSearchCV.

## Jeu de test/validation

La première étape consiste à diviser les données en un jeu de test et un jeu de validation. Cette division nous permettra d'évaluer la capacité de généralisation de nos modèles. La fonction train_test_split de scikit-learn est utilisée pour cette tâche.

```python
train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
``` 

## Visualisation des données et recherche de corrélations


La visualisation des données est cruciale pour comprendre les relations entre les variables. Nous utilisons la bibliothèque Pandas pour afficher les premières lignes du dataframe et calculer la matrice de corrélation.

```python
corr = df.corr(numeric_only=True)
``` 

La heatmap générée nous donne une vision claire des relations entre les variables. Nous avons également utilisé One-Hot Encoding pour traiter la variable catégorielle "ocean_proximity".

## Préparation des données

La préparation des données comprend plusieurs étapes telles que la séparation des variables explicatives de la variable à expliquer, la gestion des données manquantes, et la transformation des variables qualitatives.

```python
y_train = train_set['median_house_value']
x_train = train_set.drop(['median_house_value'], axis=1)
```

Nous avons choisi de supprimer les lignes avec des valeurs manquantes plutôt que de remplacer les valeurs manquantes. Cette décision dépend de la quantité de données manquantes et de la nature des données.

```python
df = df.dropna()
```
La gestion des variables qualitatives a été réalisée avec les classes LabelEncoder et OneHotEncoder.


```python
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
```
Nous avons également recalibré les variables pour qu'elles soient à la même échelle en utilisant le StandardScaler.

## Régression Linéaire

La régression linéaire est un modèle simple qui cherche à établir une relation linéaire entre les variables explicatives et la variable à expliquer. Nous avons utilisé la classe LogisticRegression de scikit-learn.



```python
linear_pipe = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', max_iter=10000))
linear_pipe.fit(x_train, y_train)
score_linear = linear_pipe.score(x_test, y_test)
```

## Arbre de Décision

Les arbres de décision sont des modèles basés sur des règles de décision. Nous avons utilisé la classe DecisionTreeRegressor de scikit-learn.

```python
decision_tree_pipe = make_pipeline(StandardScaler(), DecisionTreeRegressor())
decision_tree_pipe.fit(x_train, y_train)
score_decision_tree = decision_tree_pipe.score(x_test, y_test)
```

## Forêts Aléatoires

Les forêts aléatoires combinent plusieurs arbres de décision pour améliorer les prédictions. Nous avons utilisé la classe RandomForestRegressor de scikit-learn.


```python
random_forest_pipe = make_pipeline(StandardScaler(), RandomForestRegressor())
random_forest_pipe.fit(x_train, y_train)
score_random_forest = random_forest_pipe.score(x_test, y_test)
```

## Recherche d'un Modèle

Nous avons utilisé les classes GridSearchCV pour rechercher les meilleurs hyperparamètres de chaque modèle. Les paramètres à optimiser ont été définis dans le dictionnaire params.

```python

grid = GridSearchCV(model, params[model_name], cv=2)
grid.fit(x_train, y_train)

```

Les résultats ont été analysés et les meilleurs modèles ont été sauvegardés pour une évaluation finale.





## Conclusion

En conclusion, ce TP nous a permis de comprendre le processus complet de création de modèles de prédiction. Nous avons exploré différentes approches, préparé les données de manière adéquate, entraîné et évalué plusieurs modèles, et finalement sélectionné le meilleur modèle en utilisant la recherche d'hyperparamètres. Cela démontre l'importance de comprendre chaque étape du processus de machine learning pour obtenir des modèles performants.






