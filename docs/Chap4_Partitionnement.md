# Chapitre IV : Partitionnement

Ce chapitre est une introduction au partitionnement (ou "classification non-supervisée") : principe, mesures de performances et méthodes de base.

![En-tête chapitre IV](img/Chap4_header.png)

---

## Problème de partitionnement

### Différents types de partionnement 

#### Par partition

#### Hiérarchique

### La labélisation

### Exemple de problème

**Comment recenser les espèces de chauves-souris présentes sur le site de l'OVSQ ?**

Les chiroptérologues (spécialistes des chauves-souris), utilisent souvent l'identification acoustique pour faire un relevé des espèces de chauves-souris présentes sur un site donné.
En effet, les ultrasons émis par les chauves-souris sont caractéristiques de leur espèce, et contrairement à la capture, cette méthode n'a aucun impact sur ces animaux, qui sont protégés en France.

Situé en bordure de la forêt des Sources de la Bièvre à Guyancourt, l'Observatoire de Versailles Saint-Quentin-en-Yvelines (OVSQ) est traversé toutes les nuits d'été par des chauves-souris.
C'est pourquoi depuis 2024, l'OVSQ installe un enregistreur d'ultrasons sur son site, afin de recenser les différentes espèces présentes.

Voici quelques exemples de "sonogrammes" obtenus à partir des enregistrements de l'OVSQ :

![Exemples de sonogrammes](img/Chap4_exemple_sonogrammes.png)

Chaque image représente un cri de chauve-souris : en abscisse le temps, en ordonnée la fréquence, et en nuance de gris l'amplitude.
On a l'impression que ces 5 cris proviennent de 5 espèces différentes : fréquences moyennes différentes, plage de fréquences différentes, durées différentes, formes différentes et nombre d'harmoniques différentes.

Pour les besoins de ce cours, ont été sélectionnés 474 enregistrements de cris de chauves-souris provenant de l'OVSQ.
Nous aimerions entrainer un modèle à reconnaitre les espèces de chauves-souris enregistrées, mais nous n'avons pas de vérité terrain pour vérifier ses prédictions. 
**Est-il tout de même possible de diviser ces enregistrements en plusieurs classes selon leurs similarités, et d'identifier par la suite l'espèce correspondant à chaque classe ?**

Dans ce but, les 3 features suivantes ont été retenues pour chaque enregistrement de cri de chauve-souris : la fréquence moyenne du fondamental (kHz), l'écart-type en fréquence du fondamental (kHz), et la durée du cri (ms)

Voici le jeu de données complet, au format CSV : [Chap4_bats_dataset](https://github.com/NicOudart/UVSQ_LSSI633_data_science/tree/master/datasets/Chap4_bats_dataset.csv)

Le tableau de données qu'il contient est de la forme :

|freq_mean|freq_std|time_len|
|:-------:|:------:|:------:|
|31.000   |3.566   |7.500   |
|30.340   |3.068   |6.250   |
|28.921   |2.278   |4.750   |
|27.218   |2.136   |9.750   |
|29.574   |4.091   |6.750   |
|...      |...     |...     |
|26.605   |4.166   |4.750   |
|23.630   |6.046   |5.750   |
|26.000   |3.716   |4.500   |



## Mesures de performances

### Inertie intra-classe et internie inter-classe

### Coefficient de silhouette

### La méthode du coude

## Méthodes de base

### K Moyennes

### Classification Ascendante Hiérarchique