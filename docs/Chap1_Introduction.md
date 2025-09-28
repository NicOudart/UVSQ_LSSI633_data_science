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
|La bibliothèque Scikit-Learn possède dans son package "preprocessing" une fonction "OrdinalEncoder", permettant d'assigner un entier à des variables qualitatives ordinales.|
|Dans ce même package, vous trouverez également une fonction "OneHotEncoder", permettant d'encoder en one-hot des variables qualitatives nominales.|
|Dans les 2 cas, il vous faut créer une instance de "OrdinalEncoder" ou de "OneHotEncoder", puis utiliser la méthode "fit_transform()" avec vos données en entrée.|

Les variables étudiées peuvent aussi être **multidimensionnelles**.


Enfin, les données étudiées peuvent être **structurées**.



### Visualisation graphique

### Statistiques descriptives

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