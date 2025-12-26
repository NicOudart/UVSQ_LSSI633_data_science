# Chapitre III : Régression

Ce chapitre est une introduction à la régression : principe, mesures de performances et méthodes de base.

![En-tête chapitre III](img/Chap3_header.png)

---

## Problème de régression

Comme mentionné lors du Chapitre I, par "**régression**" on entend associer une réalisation d'une variable **quantitative continue** à un individu (labels), à partir des réalisations d'autres variables (features).

On cherche donc une **relation entre les variables** d'entrée (aussi appelées "variables explicatives") et la variable de sortie (aussi appelée "variable de réponse").

Il existe 2 grands types de relations entre variables :

* La relation est **déterministe** si on considère que la variable de sortie peut être **connue exactement** à partir des variables d'entrée.

Par exemple, si je veux déterminer le rayon $R$ d'une étoile par rapport à sa luminosité $L$ et sa température $T$, en m'appuyant sur la théorie du corps noir, j'utilise la formule $R = \sqrt{\frac{L}{4 \pi \sigma T^4}}$.

* La relation est **probabiliste** si on considère que d'**autres facteurs** que les variables d'entrée vont influer sur la valeur de la variable de sortie.

Dans ce cas, une même réalisation des variables d'entrée pourra être associée à plusieurs valeurs de la variable de sortie, et inversement.

Mais si les variables d'entrée et de sortie sont **corrélées**, la relation sera tout de même utile pour réaliser des **prédictions** avec une certaine marge d'erreur.

C'est souvent le cas en Physique, lorsque l'on réalise des mesures pour expliquer un phénomène.
Si on reprend notre exemple précédent : la théorie du corps noir n'explique pas parfaitement la rayonnement d'une étoile, et on peut avoir des erreurs de mesures de $L$, $T$ et $R$.
Mais si $L$, $T$ et $R$ sont correlées, alors on peut essayer de prédire $R$ à partir de $L$ et $T$, moyennant une certaine erreur.

C'est tout le principe de la régression : **déterminer une relation probabiliste entre les variables d'entrée et la variable de sortie**.

Nous avions aussi mentionné lors du Chapitre I qu'entrainer un modèle de régression ne peut se faire que par **apprentissage supervisé**.
L'idée est encore une fois que le modèle soit ensuite capable de **généraliser** à de nouvelles observations.

|Anecdote historique|
|:-|
|Le nom de "régression" vient du généticien Francis Galton, qui l'utilisa pour son étude sur la "régression vers la moyenne".|
|En fait, Galton cherchait à modéliser un phénomène d'hérédité qu'il observait dans la population humaine :|
|La taille des fils avait tendance à se rapprocher de la moyenne de la population par rapport à celle de leur père.|

### Les différents types de régression

Suivant le nombre d'entrées et le type de modèle à ajuster aux données, on a différents types de problèmes de régression.
Nous allons voir ici les 3 plus courants.

#### Régression linéaire simple

La **régression linéaire simple** est le problème de régression le plus basique qui soit.

On recherche une relation linéaire entre une variable d'entrée $x$ et une variable de sortie $y$.
L'erreur est une variable aléatoire notée $\epsilon$.

Le modèle à ajuster est le suivant :

$y = a x + b + \epsilon$

avec les paramètres $a$ et $b$ à déterminer.

Il s'agit d'un problème d'**inférence statistique** : nous disposons d'un jeu d'entrainement qui est un **échantillon** de la population totale, et nous voulons en déduire une estimation de $a$ et $b$ nous permettant de réaliser des prédictions.

Nous verrons que l'on fait en général les hypothèses suivantes sur $\epsilon$ :

* **Indépedance** de ses valeurs.

* **Moyenne nulle**.

* **Ecart-type constant**.

Afin de pouvoir donner un "intervalle de confiance" aux prédictions, on va en plus ajouter une hypothèse de **normalité** : $\espilon$ suit une loi normale.
Nous en reparlerons plus tard.

#### Régression linéaire multiple

On peut généraliser le modèle de la section précédente aux problèmes avec **plusieurs variables d'entrée**.

Si on note $x_1$, $x_2$, ..., $x_n$ les $n$ variables d'entrée, et $y$ notre variable de sortie.

Le modèle à ajuster devient :

$y = a_1 x_1 + a_2 x_2 + ... + a_n x_n + b + \epsilon$

On reconnait la formule d'un hyperplan de dimension $n$.

On peut mettre cette formule sous forme matricielle :

$y = A.X + \epsilon$

avec $A = 
      \begin{pmatrix}
      a_1\\
      a_2\\
	  \vdots\\
      a_n 
      \end{pmatrix}$
	  
et $X = 
      \begin{pmatrix}
      x_1\\
      x_2\\
	  \vdots\\
      x_n 
      \end{pmatrix}$
	  
Nous verrons dans la suite comment généraliser les méthodes aux problèmes multiples.

#### Régression polynomiale

Comment faire lorsqu'un modèle linéaire n'est pas pertinent pour représenter la relation entre nos variables ?

Afin de réaliser une **régression non-linéaire**, il y a une astuce : on cherche à ajuster un modèle **polynomial**.

Pour un ordre $k$, il aura la forme :

$y = a_1 x + a_2 x^2 + ... + a_n x^n + b + \epsilon$

Il y a alors une astuce : si on considère $x$, $x^2$, ..., $x^n$ comme $n$ variables d'entrée, alors on reconnait un problème de **régression multiple** !

On peut donc le traiter avec les mêmes méthodes qu'un problème linéaire.

Il existe d'autres techniques d'ajustement de modèles non-linéaires, que nous ne traiterons pas dans le cadre de ce cours.

### Exemple de problème

**Est-il possible de prédire l'intensité du rayonnement solaire à partir du nombre de taches solaires ?**

Les taches solaires sont des zones de "faible" température (4500 K contre 5800 K) la surface du soleil (photosphère).
Elles apparaissent lorsque l'apport de chaleur à la surface par convection est inhibé par une concentration de lignes de champs magnétique.

On sait observer les taches solaires depuis l'antiquité, et on a des mesures du nombre taches solaires depuis le XVIIème, qui varie entre 0 et 350 environ.
On l'utilise comme un marqueur de l'activité solaire depuis le milieu XIXème siècle.

Un autre marqueur connu de l'activité solaire est l'"irradiance solaire" totale ou TSI en anglais.
Il s'agit de la puissance du rayonnement solaire reçue en haut de l'atmosphère ter restre, par unité de surface.
Sa valeur est d'environ 1363 $W/m^2$, avec de légères variations suivant l'activité solaire.

La TSI est un paramètre clé en climatologie, puisqu'il correspond à l'énergie apportée par le soleil à la Terre.
Mais sa mesure n'est pas aisée : il ne peut être obtenu que par des satellites.

Voici l'évolution du nombre de taches solaires et du TSI depuis 1948 jusqu'à 2024 :

![Taches solaires et TSI en fonction de l'année](img/Chap3_exemple_taches_solaires_tsi_annees.png)

On a ici 20 points par an, avec un filtrage de moyenne glissante sur une 1/2 année, soit 1531 points au total.
Elles sont issues de l'Observatoire Royal de Belgique (SILSO, Dewitte et al. 2022).

On observe que le nombre de taches solaires comme la TSI suivent les mêmes cycles de 11 ans environ, correspondant aux cycles de l'activité solaire.
On imagine alors que la corrélation entre les 2 grandeurs doit être forte.

D'où l'idée suivante : **peut-on entrainer un modèle à estimer la TSI à partir du nombre de taches solaires ?**

Voici les données d'où sont issues les courbes précédentes, au format CSV : [Chap3_sunspots_dataset](https://github.com/NicOudart/UVSQ_LSSI633_data_science/tree/master/datasets/Chap3_sunspots_dataset.csv)

Le tableau de données qu'il contient est de la forme :

|year   |tsi     |sunspots|
|:-----:|:------:|:------:|
|1948.25|1363.743|193.667 |
|1948.30|1363.729|196.541 |
|1948.35|1363.722|205.891 |
|1948.40|1363.713|215.623 |
|1948.45|1363.729|218.060 |
|...    |...     |...     |
|2024.65|1364.015|172.536 |
|2024.70|1364.013|170.525 |
|2024.75|1364.038|171.563 |

Il contient pour chacun des 1531 points l'année (sous forme décimale), la TSI moyenne (en $W/m^2$) et le nombre de taches solaires moyen sur une fenêtre d'une 1/2 année.

Notre problème de régression sera la suivant : **prédire la TSI moyenne sur une 1/2 année à partir du nombre de taches solaires moyen sur cette même fenêtre**.

Voyons d'abord si une telle régression est possible à partir de ces données.

Une fois le fichier CSV téléchargé, il peut être importé sous Python en tant que DataFrame Pandas à partir de son chemin d'accès "input_path" :

~~~
import pandas as pd
df_dataset = pd.read_csv(input_path)
~~~

On peut alors utiliser la méthode "plot" des DataFrames pandas pour afficher la TSI en fonction du nombre de taches solaires, sous la forme d'un **nuage de points** :

~~~
df_dataset.plot(x='sunspots',y='tsi',kind='scatter',c='r',marker='+')
~~~

Voici le résultat :

![Taches solaires en fonction du TSI](img/Chap3_exemple_taches_solaires_tsi.png)

On observe comme attendu que les 2 grandeurs ont l'air **fortement corrélées**.
Cependant, on peut déjà constater que : (1) la relation n'a l'air linéaire que pour des nombres de taches solaires faibles (moins de 150-200), (2) la dispersions des points a l'air d'augmenter avec le nombre de taches solaires.

Ces observations seront importantes dans la suite.

On peut également calculer le coefficient de corrélation entre la TSI et le nombre taches solaires, en utilisant la méthode "corr" des DataFrames Pandas :

~~~
df_dataset['tsi'].corr(df_dataset['sunspots'])
~~~

On trouve un coefficient de corrélation de 0.89 environ, ce qui confirme une forte corrélation entre les variables.
Vouloir entrainer un modèle à prédire la TSI à partir du nombre de taches solaires a donc un sens.

**Il est à noter que nous avons ici grandement simplifié le problème et sa résolution pour les besoins de ce cours.**
**Une vraie stratégie de validation pour optimiser les hyperparamètres et éviter le sur-apprentissage ne sera pas appliquée**.

**L'idée est que nous verrons cet exemple plus en détails en TP.**

## Mesures de performances

Nous allons passer en revue dans cette section les principaux indicateurs de performances applicables à la régression linéaire.

### Table ANOVA

Soit un problème de régression dont la variable de sortie est notée $y$.
On veut évaluer les performances d'un modèle de régression sur un jeu de données issu de ce problème.

La moyenne des valeurs de $y$ au sein de cet échantillon est notée $\overline{y}$.
La valeur de $y$ pour le i-ème individu de cet échantillon sera notée $y_i$.

Admettons que l'on a ait déterminé un modèle de regression linéaire pour ces données.
On note $\hat{y_i}$ la prédiction de ce modèle linéaire pour le i-ème individu.

Pour juger de la qualité du modèle, on divise les écarts en 2 groupes :

* Les **écarts résiduels**, ou "résidus" : $y_i - \hat{y_i}$
Il s'agit des écarts non-expliqués par le modèle.

* Les écarts **écarts de régression**, ou "écarts expliqués" : $\hat{y_i} - \overline{y}$
Il s'agit des écarts expliqués par le modèle.

On a alors l'**écart total** :

$y_i - \overline{y} = (y_i - \hat{y_i}) - (\hat{y_i} - \overline{y})$

On met en général ces écarts sous la forme de variances, en prenant la somme des carrés des $p$ individus de cet échantillon :

* SCR : $\sum_{i=1}^{p} (y_i - \hat{y_i})^2$

* SCE : $\sum_{i=1}^{p} (\hat{y_i} - \overline{y})^2$

* SCT : $\sum_{i=1}^{p} (y_i - \overline{y})^2$

avec $SCT = SCR + SCE$

**Un modèle sera d'autant plus performant que la SCR sera faible comparée à la SCT**.

L'idée est la suivante : plus la SCE est grande comparée à la SCR, et plus le modèle **explique** $y$ à partir des entrées.

On range en général ces valeurs sous la forme d'un tableau, nommé "table ANOVA" :



### Coefficient de détermination

### Analyse visuelle des résidus

#### Normalité

#### Homoscédasticité

#### Indépendance

## Méthodes de base

### Moindres carrés ordinaire

#### Principe

#### Meilleur Estimateur Linéaire Non-biaisé (BLUE)

#### Intervalles de confiance

#### Généralisation à la régression linéaire multiple

#### Remarques

### Perceptron multicouche