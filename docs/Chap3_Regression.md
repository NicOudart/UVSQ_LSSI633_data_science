# Chapitre III : Régression

Ce chapitre est une introduction à la régression : principe, mesures de performances et méthodes de base.

![En-tête chapitre III](img/Chap3_header.png)

---

## Problème de régression

### Régression linéaire

### Régression polynomiale

### Exemple de problème

**Est-il possible de prédire l'intensité du rayonnement solaire à partir du nombre de taches solaires ?**

Les taches solaires sont des zones de "faible" température (4500 K contre 5800 K) la surface du soleil (photosphère).
Elles apparaissent lorsque l'apport de chaleur à la surface par convection est inhibé par une concentration de lignes de champs magnétique.

On sait observer les taches solaires depuis l'antiquité, et on a des mesures du nombre taches solaires depuis le XVIIème, qui varie entre 0 et 350 environ.
On l'utilise comme un marqueur de l'activité solaire depuis le milieu XIXème siècle.

Un autre marqueur connu de l'activité solaire est l'"irradiance solaire" totale ou TSI en anglais.
Il s'agit de la puissance du rayonnement solaire reçue en haut de l'atmosphère terrestre, par unité de surface.
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

## Méthodes de base

### Moindres carrés ordinaire

### Perceptron multicouche