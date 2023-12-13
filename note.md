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

--- 

**Dataframe**
Un DataFrame est une structure de données tabulaire bidimensionnelle, similaire à une feuille de calcul ou à une table de base de données relationnelle. C'est une structure de données clé de la bibliothèque pandas en Python. 

En d'autres termes, un DataFrame peut être considéré comme une collection de séries (colonnes) alignées, où chaque série peut être de type différent (entier, chaîne de caractères, etc.). Il est extrêmement flexible et efficace pour analyser et manipuler des données.

Voici quelques caractéristiques importantes d'un DataFrame :

1. **Structure Tabulaire :** Les données sont organisées sous forme de table à deux dimensions, où chaque ligne représente une observation et chaque colonne représente une variable.

2. **Indices et Colonnes :** Les lignes et les colonnes d'un DataFrame sont associées à des indices, ce qui facilite l'accès et la manipulation des données.

3. **Types de Données Mixtes :** Chaque colonne peut avoir un type de données différent (entier, flottant, chaîne de caractères, etc.).

4. **Opérations Rapides :** Les opérations sur les DataFrames, telles que le filtrage, la sélection, et les opérations mathématiques, sont optimisées pour des performances élevées.

**Matrice de corrélation**


Une matrice de corrélation est une table qui montre les coefficients de corrélation entre de nombreuses variables. Chaque cellule de la matrice représente le coefficient de corrélation entre deux variables. Cette valeur est comprise entre -1 et 1. Une valeur de 1 indique une corrélation positive parfaite, -1 indique une corrélation négative parfaite, et 0 indique l'absence de corrélation.

La matrice de corrélation est souvent utilisée pour explorer la dépendance linéaire entre les variables dans un ensemble de données. Elle est utile dans l'analyse exploratoire des données, en particulier lorsqu'on travaille avec des modèles statistiques ou de machine learning, car elle peut fournir des informations sur la force et la direction des relations entre les variables.

**Coefficient de corrélation**
Un coefficient de corrélation mesure la force et la direction d'une relation linéaire entre deux variables. Il indique à quel point les variations d'une variable sont associées aux variations de l'autre variable. Le coefficient de corrélation varie de -1 à 1, et sa signe indique la direction de la relation.

Les deux coefficients de corrélation les plus couramment utilisés sont le coefficient de corrélation de Pearson et le coefficient de corrélation de Spearman :

1. **Coefficient de Corrélation de Pearson (r) :** Mesure la corrélation linéaire entre deux variables continues. Un coefficient de +1 indique une corrélation linéaire positive parfaite, -1 indique une corrélation linéaire négative parfaite, et 0 indique une absence de corrélation linéaire.

2. **Coefficient de Corrélation de Spearman (ρ) :** Mesure la corrélation monotone entre deux variables, ce qui signifie qu'il capture également les relations non linéaires. Il est basé sur les rangs des données plutôt que sur les valeurs brutes.

En résumé, un coefficient de corrélation fournit une mesure quantitative de la force de la relation entre deux variables, mais il ne détermine pas la causalité (c'est-à-dire quelle variable cause l'autre). Il est important de noter que la corrélation ne garantit pas la causalité, et d'autres analyses sont nécessaires pour établir des relations de cause à effet.




Le One-Hot Encoding est une technique de représentation des variables catégorielles (ou nominatives) en tant que vecteurs binaires. Elle est souvent utilisée dans le domaine de l'apprentissage machine lorsque les modèles nécessitent des variables numériques en entrée. Le One-Hot Encoding est particulièrement utile pour traiter les variables catégorielles qui n'ont pas d'ordre intrinsèque.

Voici comment fonctionne le One-Hot Encoding :

1. **Création de Colonnes Binaires :** Pour chaque catégorie unique dans la variable catégorielle, une nouvelle colonne binaire est créée.

2. **Assignation de Valeurs Binaires :** Dans chaque colonne nouvellement créée, une valeur binaire est attribuée. Généralement, on utilise 1 pour indiquer la présence de la catégorie et 0 pour son absence.

3. **Élimination d'une Colonne pour Éviter la Multicollinéarité :** Pour éviter le piège de la multicollinéarité (corrélation élevée entre deux ou plus variables prédictives), une des colonnes binaires nouvellement créées est supprimée. C'est ce que fait l'option `drop_first=True` dans certaines implémentations, y compris dans le code que vous avez fourni.

Voici un exemple simple :

Supposons que nous ayons une variable catégorielle "Couleur" avec les catégories "Rouge", "Bleu" et "Vert". Le One-Hot Encoding créerait trois colonnes binaires : "Rouge", "Bleu", "Vert". Chaque ligne aurait une valeur de 1 dans la colonne correspondant à sa couleur et des zéros dans les autres colonnes.

Avant le One-Hot Encoding :
```
Couleur
------
Rouge
Bleu
Vert
Rouge
```

Après le One-Hot Encoding :
```
Rouge | Bleu | Vert
-----|------|-----
1    | 0    | 0
0    | 1    | 0
0    | 0    | 1
1    | 0    | 0
```

Cela transforme une variable catégorielle en une forme que les algorithmes d'apprentissage machine peuvent mieux comprendre et utiliser dans leurs calculs.