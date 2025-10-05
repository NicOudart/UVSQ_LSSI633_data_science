# Chapitre I : Introduction aux sciences des données

Ce chapitre porte sur les concepts et les enjeux des sciences des données.

---

## Analyse de données

### Nature et type des données

Une des difficultés rencontrées en sciences des données provient de la **grande variétés des données**.

Tout d'abord, les variables étudiées peuvent être de **nature** différente :

![Nature des données](img/Chap1_nature_donnees.png)

* Une donnée **quantitative continue** peut prendre n'importe quelle valeur numérique : par exemple, le prix d'un kilo de farine.

* Une donnée **quantitative discrète** ne peut prendre qu'un nombre fini de valeurs numériques dans un intervalle : par exemple, le nombre de pépites de chocolats dans une brioche.

* Une donnée **qualitative nominale** est descriptive sans ordre hiérarchique : par exemple, la région d'origine d'une pâtisserie.

* Une donnée **qualitative ordinale** est descriptive avec un ordre hiérarchique : par exemple, le niveau de cuisson d'une baguette de pain (blanche, pas trop cuite, bien cuite).

La plupart des modèles d'apprentissage automatique ne prennent que des valeurs numériques en entrée.

On va donc en général **encoder** des données qualitatives avec des **valeurs numériques**. Par exemple :

|Cuisson du pain|Encodage|
|:-------------:|:------:|
|Blanc          |1       |
|Pas trop cuit  |2       |
|Bien cuit      |3       |

Cette méthode fonctionne bien pour des données ordinales comme la cuisson du pain, mais pour des données nominales le modèle risque de croire qu'il y a un ordre hiérarchique dans les données qui n'existe pas.
C'est pourquoi on utilise souvent l'encodage **one-hot**.

L'idée est de faire comme si chaque nom possible pour une variable qualitative était une variable en soit. 
On appelle parfois ces variables imaginaires des "dummy variables".

Par exemple, pour la région d'origine des pâtisseries, on passe de :

|Pâtisserie   |Région   |
|:-----------:|:-------:|
|Croissant    |Paris    |
|Merveilleux  |Nord     |
|Kouign-amann |Bretagne |
|Cannelé      |Sud-Ouest|
|Kougelhopf   |Est      |

à l'encodage one-hot suivant :

|Pâtisserie  |Paris|Nord|Bretagne|Sud-Ouest|Est|
|:----------:|:---:|:--:|:------:|:-------:|:-:|
|Croissant   |1    |0   |0       |0        |0  |
|Merveilleux |0    |1   |0       |0        |0  |
|Kouign-amann|0    |0   |1       |0        |0  |
|Cannelé     |0    |0   |0       |1        |0  |
|Kougelhopf  |0    |0   |0       |0        |1  |

Pour le Merveilleux, on donnera donc en entrée d'un modèle le binaire 01000.

On remarque ici que plus la variable a de noms possibles, et plus les binaires d'encodage one-hot seront longs, ce qui peut être problématique.

|Astuce Python|
|:-|
|La bibliothèque Scikit-Learn possède dans son package **preprocessing** une fonction **OrdinalEncoder**, permettant d'assigner un entier à des variables qualitatives ordinales.|
|Dans ce même package, vous trouverez également une fonction **OneHotEncoder**, permettant d'encoder en one-hot des variables qualitatives nominales.|
|Dans les 2 cas, il vous faut créer une instance de **OrdinalEncoder** ou de **OneHotEncoder**, puis utiliser la méthode **fit_transform()** avec vos données en entrée.|

Les variables étudiées peuvent aussi être **multidimensionnelles**.

En effet, dans la pluplart des situations, notre jeu de données peut se mettre sous la forme d'un tableau, dont 

* Les colonnes correspondront aux "**variables**".

* Les lignes correspondront aux "**individus**" : les différentes réalisations de ces variables.

L'ensemble des individus sera nommé "**population**", une sélection des individus un "**échantillon**".

Voici un exemple de jeu de données multidimensionnelles :

|Brioche n°1|Poids (g)|Nombre de pépites de chocolat|Prix (€)|
|:---------:|:-------:|:---------------------------:|:------:|
|1          |70       |13                           |3.5     |
|2          |80       |17                           |3.6     |
|3          |85       |15                           |3.7     |
|4          |83       |16                           |3.4     |
|5          |76       |18                           |3.3     |
|6          |78       |13                           |3.5     |

Nous avons ici 6 individus, les brioches, pour lesquelles nous avons mesuré 3 variables, le poids, le nombre de pépites de chocolat, et le prix.

|Astuce Python|
|:-|
|Pour stocker puis manipuler des données multidimensionnelles, on utilise souvent en Python un type de conteneur de la bibliothèque **Pandas** : les "**DataFrames**".|
|Les DataFrames se présentent comme des tableaux pouvant contenir des variables de types différents, avec un label associé à chaque colonne du tableau (variable).|
|Nous reparlerons de Pandas plus loin dans ce chapitre.|

Enfin, les données étudiées peuvent être **structurées**.

On entend par là que des données peuvent avoir un cohérence chronologique (série temporelle, un son) ou spatiale (une carte, une image, un texte, une vidéo).

Par exemple, dans le cas d'une image :

![Données structurées](img/Chap1_image.png)

Chaque pixel de l'image doit être compris dans le contexte global de l'image. 
Il est évident que changer la position des pixels les uns par rapport aux autres change le jeu de données : 

![Données structurées](img/Chap1_image_shuffled.png)

Dans certains cas, l'**ordre des données** est donc en soit une information nécessaire à leur interprétation.

Vous l'aurez compris, la nature des données, leur dimensionnalité, ainsi que leur structure, **peuvent rendre leur compréhension difficile**.
Nous allons dans la suite voir comment on peut essayer de tirer des informations pertinentes de nos données.

### Visualisation graphique

La 1ère étape lorsque l'on cherche à comprendre ses données, c'est d'essayer de les **visualiser** de manière pertinente.
Nous allons voir les types de **représentations graphiques** les plus classiques pour visualiser un jeu de données.

#### Courbes et nuages de points :



#### Diagrammes en barres et histogrammes :



#### Boîtes à moustaches :



#### Kernel Density Estimation (KDE) :



#### Graphique en aires :



#### Diagramme circulaire (camembert) :



#### En Python

|Astuce Python|
|:-|
|La bibliothèque Python "Pandas", dont nous reparlerons plus tard dans ce chapitre, propose une méthode "plot" qui permet des affichages graphiques à partir de jeux de données.|
|Il suffit donner le bon paramètre "kind" en entrée pour obtenir le type d'affichage voulu :|
|- "line" : une courbe.|
|- "scatter" : un nuage de points.|
|- "bar" : un diagramme en barres vertical.|
|- "barh" : un diagramme en barres horizontal.|
|- "hist" : un histogramme.|
|- "box" : des boîtes à moustaches.|
|- "kde" : une "kernel density estimation".|
|- "area" : un graphique en aires.|
|- "pie" : un diagramme circulaire.|

### Statistiques descriptives

Toujours dans l'objectif de comprendre notre jeu de données, on peut essayer de **décrire** chaque variable par des **indicateurs statistiques**.
Nous allons voir ici les indicateurs les plus communs en statistiques descriptives.

#### Moyenne, médiane et mode



#### Variance et écart-type



#### Quantiles



#### Asymétrie et kurtosis



### Recherche de corrélation

### Préparation des données

## Les apprentissages

### L'apprentissage automatique

### Les 3 grands types d'apprentissages

### Entraînement d'un modèle

## Enjeux de l'apprentissage

### Quantité et qualité des données

### Représentativité et équilibre des données

### Pertinence des variables

### Sur-apprentissage / sous-apprentissage

## Test, validation et hyperparamètres

## Import de données et fichiers CSV

## Outils Python pour l'apprentissage

### Pandas

### Scikit-Learn

### Keras-Tensorflow, Pytorch