# Chapitre IV : Partitionnement

Ce chapitre est une introduction au partitionnement (ou "classification non-supervisée") : principe, mesures de performances et méthodes de base.

![En-tête chapitre IV](img/Chap4_header.png)

---

## Problème de partitionnement

Comme mentionné lors du Chapitre I, par "**partitionnement**" on entend diviser des individus non-labélisés en groupes suivant leur proximité dans l'espace des features.
On essayera d'assigner des labels à ces groupes par la suite.

L'idée est de **mieux comprendre** un jeu de données, et d'essayer de **classifier de nouvelles données**.

### Différents types de partitionnement

Suivant le problème à résoudre, il existe 2 grands types de partitionnement : **par partition** et **hiérarchique**.

On parle de partitionnement "**par partition**" lorsque l'on cherche à diviser les individus d'une base de donnée en un nombre fini de groupes $k$, sans tisser de lien entre eux.
On a alors aucune information sur la proximité des classes entre elles.

Par opposition, on parle de partitionnement "**hiérarchique**" lorsque l'on va diviser les individus d'une base de données en $k$ groupes, hiérarchisés selon leur similarité.
Cette hiérarchisation sera dite "**descendante**" ou "**ascendante**" suivant si on part de 1 classe vers $k$ classes, ou inversement.

On représente souvent la hiérarchisation des classes sous la forme d'un diagramme appelé **dendrogramme** : un arbre représentant les liens entre classes (en abscisse) et leurs distances (en ordonnée).



### La labélisation

Une fois les données séparées en $k$ groupes, l'étape suivante est souvent d'essayer d'attribuer des labels aux classes ainsi déterminées.
Ceci permettra de donner une interprétation à notre partition, et à nos futures prédictions.
On appelle ce processus "**labélisation**".

En l'absence de vérité terrain pour chacun des individus, la labélisation est délicate.
On peut néanmoins proposer la méthode suivante :

* Essayer de caractériser chaque groupe avec les outils de **statistiques descriptives** vus au Chapitre 1 (moyenne, écart-type, etc.).

* Si une partition hiérarchique a été réalisée, étudier aussi les **liens entre les groupes**. Sinon, analyser les distances entre groupes.

* Comparer les caractéristiques de chaque groupe, ainsi que les liens entre groupes, aux **connaissances établies** sur le domaine d'application, ou à un petit échantillon de données labélisées si disponible.

* Attribuer un label à chaque groupe en se basant sur ces éléments.

La labélisation implique donc une certaine **expertise** dans le domaine où on cherche à appliquer de la classification non-supervisée.

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

Dans ce but, les 3 features suivantes ont été retenues pour chaque enregistrement de cri de chauve-souris : la fréquence moyenne du fondamental (kHz), l'écart-type en fréquence du fondamental (kHz), et la durée du cri (ms).

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

Notre problème de partitionnement sera le suivant : **Identifier les différentes espèces de chauves-souris dans les enregistrements de l'OVSQ, à partir de la fréquence moyenne du fondamental, de l'écart-type en fréquence du fondamental, et de la durée du cri**.

Assurons-nous d'abord qu'une telle partition est possible à partir de ces données.

Une fois le fichier CSV téléchargé, il peut être importé sous Python en tant que DataFrame Pandas à partir de son chemin d'accès "input_path" :

~~~
import pandas as pd
df_dataset = pd.read_csv(input_path)
~~~

Il est possible avec Seaborn d'afficher ces données sous la forme d'une **matrice de corrélations**, avec des densités estimées par **KDE** 2D :

~~~
import seaborn as sns
sns.pairplot(df_dataset,kind='kde')
~~~

Voici le résultat :

![Matrice de corrélations des enregistrements de chauves-souris de l'OVSQ](img/Chap4_correlation_matrix_kde_bats.png)

On observe que ces 3 features font apparaitre différents regroupements d'enregistrements : les distributions sont clairement multimodales.
Si les frontières entre groupes, ainsi que le nombre exact de groupes restent difficiles à établir, il n'y a aucun doute sur la présence de plusieurs groupes.
Et ces classes sont probablement liées à l'espèce de chauves-souris.

Essayer de partitionner notre base de données à partir de ces features a donc du sens.
Restera alors à labéliser les classes ainsi délimitées.

Cependant, on peut noter que certains groupes visibles ont l'air moins denses que d'autres.
Ceci est plausible : on imagine bien que certaines espèces sont plus communes sur le site que d'autres.
Un tel déséquilibre pourrait être problématique pour entrainer notre modèle.

**Il est à noter que nous avons ici grandement simplifié le problème et sa résolution pour les besoins de ce cours.**
**Une vraie stratégie de validation pour optimiser les hyperparamètres et éviter le sur-apprentissage ne sera pas appliquée**.

**L'idée est que nous verrons cet exemple plus en détails en TP.**

## Mesures de performances

Un des grands problèmes en classification non-supervisée est que le nombre de classes est une entrée de la plupart des méthodes de résolution.
**Mais comment connaitre le nombre de classes pertinentes pour un jeu de données ?**

Il faut tester différents nombres de classes plausibles, et évaluer les performances du modèle obtenu pour chacun.

Problème : les données auxquelles on veut appliquer une méthode de partitionnement étant par définition non-labélisées, **on ne peut pas calculer une erreur par rapport à une vérité terrain**.
Il existe néanmoins des critères pour évaluer la pertinence d'un partitionnement.

Nous allons voir dans cette section différents **critères pour évaluer un partitionnement**, et différentes méthodes pour **déterminer un nombre de classes optimal** pour un jeu de données.

### Inertie intra-classe et internie inter-classe

Un bon partitionnement a les 2 caractéristiques suivantes :

* **Les individus au sein d'un groupe sont les plus similaires possibles** (leurs distances dans l'espace des features sont les plus faibles possibles).

* **Les différents groupes sont les plus différents possibles** (leurs distances dans l'espace des features sont les plus grandes possibles).

On utilise souvent comme indicateurs de ces 2 caractéristiques l'**inertie** intra-classe et inter-classe.

L'**inertie d'une classe** $i$ contenant $n_i$ individus est définie comme la somme des distances au centre de gravité $g_i$ de la classe :

$I_i = \sum_{j=1}^{n_i} d(x_{i,j},g_i)^2$

où chaque $x_{i,j}$ est un vecteur contenant les réalisations des différentes features pour un individu de la classe $i$.

Il s'agit d'une analogie avec la notion de moment d'inertie en Physique : la répartition de la masse dans un objet autour de son centre de gravité va rendre plus ou moins difficile sa mise en mouvement.
D'une manière analogue, la répartition des individus dans un groupe va rendre plus ou moins coûteuse en termes de performances un changement de centre de gravité des groupes (idem pour les groupes vis-à-vis du centre de gravité du jeu de données total).

Cette formule dépend bien évidemment de la définition du **centre de gravité** $g_i$ de la classe $i$, et de la **mesure de distance** $d$ choisie.

Pour le centre de gravité, on va souvent considérer le **barycentre** :

$g_i = \frac{1}{n_i} \sum_{j=1}^{n_i} x_{i,j}$

Pour les mesures de distances, reportez-vous à la section "K plus proches voisins" du Chapitre 2.
Comme pour les K plus proches voisins, il s'agira d'un **hyperparamètre à optimiser**.

On définit alors l'**inertie intra-classe** comme étant la somme des inerties des $k$ classes :

$I = \sum_{i=1}^{k} I_i = \sum{i=1}^{k} \sum_{j=1}^{n_i} d(x_{i,j},g_i)^2$

Il s'agit d'un indicateur de la **similarité des individus au sein de chaque classe**.

L'**inertie inter-classe** est quant à elle définie comme :

$J = \sum_{i=1}^{k} n_i d(g_i,g)^2$

avec $g = \frac{1}{\sum_{i=1}^{k} n_i} \sum_{i=1}^{k} \sum_{j=1}^{n_i} x_{i,j}$ le barycentre du jeu de données complet.

Il s'agit d'un indicateur de la **séparabilité des différentes classes**.

|Théorème de Huygens|
|:-|
||

### Coefficient de silhouette

### La méthode du coude

## Méthodes de base

### K Moyennes

### Classification Ascendante Hiérarchique

## Labélisation de l'exemple

![Exemple labélisé par Tadarida](img/Chap4_exemple_labelise.png)