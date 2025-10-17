# Chapitre I : Introduction aux sciences des données

Ce chapitre porte sur les concepts et les enjeux des sciences des données.

---

## Analyse de données

### Nature et type des données

Une des difficultés rencontrées en sciences des données provient de la **grande variétés des données**.

#### Données de différentes natures

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

#### Données multidimensionnelles

Les données étudiées peuvent aussi être **multidimensionnelles**.

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

#### Données structurées

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

Il est important de savoir comment ces indicateurs sont définis afin de comprendre les informations qu'ils donnent ou ne donnent pas sur un jeu de données.

#### Moyenne, médiane et mode

Lorsque l'on veut connaitre l'ordre de grandeur des valeurs d'une variable, là où se rassemblent la plupart des valeurs, on va utiliser un indicateur de **tendance centrale** : moyenne, médiane ou mode.

* Il existe plusieurs façon de définir la moyenne, mais la plus connue est la **moyenne arithmétique** :

$\overline{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$

On note en effet souvent $\overline{x}$ la moyenne d'une variable $x$.

* La **médiane** est la valeur séparant les valeurs de la variable en 2 groupes de même taille : la moitié des valeurs sont supérieures à la médiane, l'autre moitié lui sont inférieures.

* Le **mode** est la valeur la plus représentée dans l'ensemble des valeurs de la variable.

|Astuce Python|
|:-|
|Dans la bibliothèque Python "Pandas", dont nous reparlerons plus tard dans ce chapitre, il y a ces méthodes associées aux objets DataFrames :|
|- ".mean()": la moyenne.|
|- ".median()": la médiane.|
|- ".mode()": le mode.|

#### Variance et écart-type

Lorsque l'on veut savoir à quel point les valeurs d'une variable fluctuent autour de la valeur centrale, on va utiliser des **indicateurs de dispersion**.

* Les valeurs extrêmes de la variable, le **min** et le **max**, pour connaitre l'étendue de la variable.

* La **variance** est définie par la moyenne des carrées des écarts à la moyenne :

$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \overline{x})^2$

* L'**écart-type** (souvent noté $\sigma$) est la racine carrée de la variance, soit a moyenne quadratique de écarts à la moyenne. 
Contrairement à la variance, il a l'avantage d'être **homogène à la variable étudiée**.

|Astuce Python|
|:-|
|Dans la bibliothèque Python "Pandas", dont nous reparlerons plus tard dans ce chapitre, il y a ces méthodes associées aux objets DataFrames :|
|- ".min()" et ".max()": le minimum et le maximum.|
|- ".var()": la variance.|
|- ".std()": l'écart-type.|

#### Quantiles

Afin d'avoir plus d'informations sur la répartition de valeurs d'une variable, on peut généraliser la notion de médiane en utilisant ce que l'on appelle les **quantiles** :
La division des valeurs de la variables en groupes de tailles égales.

* **Quartiles** : 3 indicateurs en divisant les valeurs de la variable en 4 groupes (25%,50% et 75%).

* **Déciles** : 9 indicateurs en divisant les valeurs de la variable en 10 groupes (10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%).

* **Centiles** : 99 indicateurs en divisant les valeurs de la variable en 100 groupes (1%, 2%, 3%, ..., 98%, 99%).

#### Asymétrie et kurtosis

Enfin, si l'on veut une information sur la répartition des valeurs d'une variable, sous la forme d'un indicateur unique, on va utiliser un **indicateur de forme**.

* Le coefficient d'**asymétrie** ("skewness" en anglais, souvent noté $\gamma_1$) permet de quantifier le désequilibre de la répartition des valeurs de la variable de chaque côté de sa valeur centrale. 

$\gamma_1 = \frac{1}{N \sigma^3} \sum_{i=1}^{N} (x_i - \overline{x})^3$

Un coefficient négatif indique un décalage à droite, un coefficient positif un décalage à gauche, et un coefficient nul une distribution symétrique.

NB : Pour la loi normale, on a $\gamma_1 = 0$.

* Le **kurtosis** (souvent noté $\gamma_2$) permet de quantifier l'acuité ou l'applatissement de la répartition des valeurs de la variable autour de sa valeur centrale.

$\gamma_2 = \frac{1}{N \sigma^4} \sum_{i=1}^{N} (x_i - \overline{x})^4$

Un kurtosis positif est un indicateur de valeurs anormales de la variable (aux extrêmités) plus fréquentes.
Un kurtosis négatif est un indicateur d'une distribution très applatie des valeurs de la variable.

NB : Pour la loi normale, on a $\gamma_2 = 0$.

### Recherche de corrélation

Une fois que l'on a décrit statistiquement les différentes variables de notre jeu de données, on va souvent vouloir essayer des tisser des liens entre ces variables.
Cette **analyse exploratoire** des données a 2 principales utilités :

* Voir si une ou plusieurs variables pourraient servir à en **prédire** une ou plusieurs autres.

* Essayer de **réduire la dimensionnalité** d'un problème basé sur ces variables.

En effet, comme évoqué précédemment, les jeux de données sont souvent multidimensionnels.
Quand la dimension des données devient très grande, la quantité de données devient peu dense en comparaison, ce qui rend difficile leur interprétation.
On appelle communément ce problème le "Fléau de la dimension" ("curse of dimensionality" en anglais).

Nous allons voir dans un 1er temps comment essayer de déterminer ce que l'on appelle des "**corrélations**" entre variables.
Puis nous verrons une méthode classique de réduction de dimension appelée "**Analyse en Composantes Principales**".

#### Matrice de corrélation

Une 1ère approche pour essayer de tisser des liens ou "**corrélations**" entre les variables et de tracer ce que l'on appelle une matrice de nuages de points, ou "**scatter-matrix**" en anglais.

L'idée est d'afficher une **matrice de graphiques**, représentant chacun **une variable en fonction d'une autre**, sous la forme d'un nuage de points.
La diagonale n'étant pas très utile (une variable en fonction d'elle-même), on la remplace en général par un histogramme de la variable en question.

Ce type de représentation permet de détecter visuellement des **relations entre les variables**.

Voici un exemple :



|Astuce Python|
|:-|
|Dans la bibliothèque Python "Pandas", dont nous reparlerons plus tard dans ce chapitre, il y a une méthode "plotting.scatter_matrix()", qui permet d'afficher une "scatter_matrix".|

Pour quantifier la corrélation entre 2 variables $x$ et $y$, on va souvent se contenter de mesurer à quel point une **relation linéaire** $y = a x + b$ peut être tirée de ces variables.
Pour cela, on va calculer le **coefficient de corrélation** de Pearson :

$r = \frac{\sum_{i=1}^{N} (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=1}^{N} (x_i - \overline{x})^2 \sum_{i=1}^{N} (y_i - \overline{y})^2}}$

La valeur de ce coefficient est toujours compris entre -1 et 1 : 

* Une valeur de 1 signifie une **corrélation** parfaite entre les variables.

* Une valeur de -1 signifie une **anti-corrélation** parfaite entre les variables.

* Une valeur de 0 signifie une **décorrélation** parfaite entre les variables : elles sont **indépendantes**.

Il y a du sens à vouloir prédire une variable à partir d'une autre si elles sont corrélées / anti-corrélées.
On peut aussi imaginer réduire la dimensionnalité d'un problème s'il se base sur plusieurs variables qui ne sont pas indépendantes.

NB : **Attention !** Corrélation entre variables n'implique pas causalité entre variables !

On affiche souvent les coefficients de corrélation obtenus pour toutes les combinaisons de variables possibles sous la forme d'une matrice : la **matrice de corrélation** de ces variables.
La diagonale de la matrice ne contient bien évidemment que des 1.

Voici un exemple :



|Astuce Python|
|:-|
|Dans la bibliothèque Python "Pandas", dont nous reparlerons plus tard dans ce chapitre, il y a une méthode "corr()" associée aux DataFrames.|
|Elle retourne une matrice de corrélation du jeu de données.|

#### Analyse en Composantes Principales (ACP)

Comme mentionné précédemment, les jeux de données que l'on rencontre sont souvent multidimensionels.
Ceci rend difficile voir impossible un affichage graphique compréhensible des individus d'une variable par rapport à une autre (il faudrait un graphique 2D pour 2 variables, 3D pour 3 variables, 4D pour 4 variables, etc.).

Afin de représenter des données multidimensionnelles sous la forme d'un affichage graphique de **dimension faible** (en général 1, 2 ou 3), on utilise souvent l'**Analyse en Composantes Principales** (ACP).

L'idée est la suivante.
Soit un jeu de données contenant $p$ variables et $n$ individus. 
On va chercher $q$ nouvelles variables par **projections linéaires** des $p$ variables d'origine, avec $q < p$, de manière à **perdre le moins d'information possible** sur le jeu de données.

Ces $q$ nouvelles variables sont alors nommées **composantes principales**.

Il existe plusieurs algorithmes pour obtenir ce résultat, celui implémenté dans la bibliothèque Python Scikit-Learn se base sur la Décomposition en Valeurs Singulières (SVD) de la matrice de données :

$X = 
\begin{pmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,p} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,p} \\
\vdots  & \vdots  & \ddots & \vdots  \\
x_{n,1} & x_{n,2} &\cdots & x_{n,p} 
\end{pmatrix}$

où chaque colonne correspond à une variable, et chaque ligne correspond à un individu.

On peut voir l'ACP comme le choix du sous-espace de dimension $q$ tel que le nuage de points projetés ait la variance la plus grande possible.

Les résultats d'une ACP peuvent être affichés sous la forme d'un **nuage de points 2D ou 3D** ($q = 2$ ou $3$) représentant les différents individus, avec pour axes les composantes principales.

Voici un exemple :



L'idée est de voir si on peut séparer les individus en différents groupes à partir des composantes principales.

Pour juger de la qualité d'une ACP, on utilise un type de graphique appelé "**cercle des corrélations**".
Ce graphique 2D représente sur chaque axe la corrélation des $p$ variables d'origine avec les $q$ composantes principales.
Chacune des $p$ variables correspond à un vecteur sur ce graphique, et un cercle de rayon 1 est également affiché pour comparaison.

Voici un exemple :



Un cercle des corrélations permet donc de juger de la corrélation des variables d'origines avec les composantes principales, et de la corrélation des variables d'origine entre elles :

* Plus une variable d'origine est proche du cercle, plus elle est fidèlement représentée par l'ACP. Dans l'idéal, on voudrait donc que toutes les variables soient proches du cercle.

* Pour 2 variables d'origine proches du cercle, si l'angle entre 2 les variables est aigu elles sont corrélées, s'il est obtu elles sont anti-corrélées, et s'il est droit elles sont décorrélées.

On pourra utiliser la projection des données renvoyée par l'ACP pour entrainer des modèles d'apprentissage.

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