# Chapitre III : Régression

Ce chapitre est une introduction à la régression : principe, mesures de performances et méthodes de base.

![En-tête chapitre III](img/Chap3_header.png)

---

## Problème de régression

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

On a ici 20 points par an, avec un filtrage de moyenne glissante sur une 1/2 année, soit 1521 points au total.
Elles sont issues de l'Observatoire Royal de Belgique (SILSO, Dewitte et al. 2022).

On observe que le nombre de taches solaires comme la TSI suivent les mêmes cycles de 11 ans environ, correspondant aux cycles de l'activité solaire.
On imagine alors que la corrélation entre les 2 grandeurs doit être forte.

D'où l'idée suivante : **peut-on entrainer un modèle à estimer la TSI à partir du nombre de taches solaires ?**

Voici les données d'où sont issues les courbes précédentes, au format CSV : [Chap3_sunspots_dataset](https://github.com/NicOudart/UVSQ_LSSI633_data_science/tree/master/datasets/Chap3_sunspots_dataset.csv)



![Taches solaires en fonction du TSI](img/Chap3_exemple_taches_solaires_tsi.png)

## Mesures de performances

## Méthodes de base

### Moindres carrés

### Perceptron multicouche