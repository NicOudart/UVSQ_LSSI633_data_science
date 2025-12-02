# Chapitre II : Classification supervisée

Ce chapitre est une introduction à la classification supervisée : principe, mesures de performances et méthodes de base.

![En-tête chapitre II](img/Chap2_header.png)

---

## Problème de classification

### Principe de la classification supervisée

Comme mentionné lors du Chapitre I, par "**classifier**" on entend associer une réalisation d'une variable **quantitative discrète** ou **qualitative** à un individu (labels), à partir des réalisations d'autres variables (features).



### Les différents types de classification

#### Binaire

#### Multi-classes

#### Multi-étiquettes

#### Multi-sorties

### Exemple de problème

**Pourquoi est-on capables de reconnaitre le son d'un instrument de musique d'un autre ?**

Lorsqu'un instrument joue une note, le son émit ne contient jamais qu'une seule fréquence.
Il est en réalité constitué d'une "fréquence fondamentale" (la note que l'on veut jouer), et des "harmoniques" (des fréquences multiples de la fondamentale).

Pour une même note jouée, suivant l'instrument, les harmoniques n'auront pas la même amplitude comparée à la fondamentale.
C'est ce que l'on appelle le "timbre" de l'instrument.
Lorsque nous écoutons de la musique, et que nous reconnaissons le son d'un instrument, c'est grâce à son timbre.

Voici 3 exemples de spectres issus d'enregistrements d'une flute, d'un hautbois et d'une trompette jouant un La (440 Hz) :

![Spectres des 3 instruments](img/Chap2_spectres_instruments.png)

On voit nettement la différence de timbre entre les 3 instruments.

D'où l'idée suivante : **peut-on entrainer un modèle à reconnaitre un instrument à partir d'un enregistrement ?**

Voici un jeu de données au format CSV, collectées à partir de milliers d'enregistrements d'une flute, d'un hautbois et d'une trompette jouant un La (440 Hz) : [Chap2_instruments_dataset](https://github.com/NicOudart/UVSQ_LSSI633_data_science/tree/master/datasets/Chap2_instruments_dataset.csv)

Le tableau de données qu'il contient est de la forme suivante :

|instrument|harmo1 |harmo2 |harmo3 |
|:--------:|:-----:|:-----:|:-----:|
|oboe      |11.842 |11.58  |10.28  |
|flute     |-17.083|-17.384|-21.496|
|trumpet   |-8.152 |-24.089|-23.813|
|oboe      |9.381  |12.434 |11.905 |
|oboe      |-1.217 |2.082  |16.275 |
|trumpet   |-3.294 |-13.812|-17.934|
|trumpet   |-4.118 |-13.485|-18.985|
|...       |...    |...    |...    |
|trumpet   |-7.762 |-5.934 |-23.308|
|flute     |-17.96 |-19.406|-22.409|
|oboe      |7.764  |6.618  |13.361 |

Il contient pour chacun des 5612 enregistrements le nom de l'instrument, et l'amplitude en dB des 3 premières harmoniques relativement à la fondamentale.

Notre problème de classification sera le suivant : **prédire l'instrument ayant joué un La à partir des amplitudes des 3 premières harmoniques**.

Voyons d'abord si une telle classification est possible à partir de ces données.

Une fois le fichier CSV téléchargé, il peut être importé sous Python en tant que DataFrame Pandas à partir de son chemin d'accès "input_path" :

~~~
df_dataset = pd.read_csv(input_path)
~~~

Il est possible avec Seaborn d'afficher ces données sous la forme d'une **matrice de corrélations**, avec chaque classe d'une couleur différente.
Ce type de représentation permet de vérifier la séparabilité des différentes classes à partir des features sélectionnés.

Voici la commande Seaborn :

~~~
sns.pairplot(df_dataset,hue='instrument')
~~~

On obtient alors le graphique suivant :

![Matrice de corrélations des 3 instruments](img/Chap2_correlation_matrix_instruments.png)

On observe que les classes "flute", "oboe" et "trumpet" sont plutôt bien séparables à partir des amplitudes des 3 premières harmoniques.
Vouloir entrainer un modèle à reconnaitre un de ces instruments à partir de ces données à donc du sens.

**Il est à noter que nous avons ici grandement simplifié le problème et sa résolution pour les besoins de ce cours.**
**Nous verrons cet exemple plus en détails en TP.**

## Mesures de performance

### Matrice de confusion

### Précision / rappel

### Courbe ROC

## Méthodes de base

### Décision Bayesienne

La **décision Bayesienne**, aussi connue sous le nom de "classification Bayesienne naïve" est une méthode de classification se basant sur un **modèle probabiliste** des features, considérées **indépendantes**, et du **théorème de Bayes**.

#### Principe

Imaginons que nous avons un problème de classification avec $q$ **classes** $C_1$, $C_2$, ..., $C_q$.
Nous voulons prédire la classe à laquelle appartient un individu.

La probabilité de chaque classe $i$ notée $p(C_i)$, aussi appelée "**probabilité a priori**".

On a $\sum_{i=1}^{q} p(C_i) = 1$ et on peut facilement estimer les différents $p(C_i)$ à partir du nombre d'occurences de $C_i$ dans les données divisée par la taille de la base de données.

En ne connaissant que les probabilités a priori de chaque classe, nous serions obligés de classer n'importe quel individu comme appartenant à la classe $C_i$ ayant le $p(C_i)$ le plus élevé.
Nous aurions alors un classifieur retournant toujours la même classe. 
Pas très utile...

Or, nous avons en réalité accès à plus d'informations : nos fameuses "features", dont nous voulons nous servir pour prédire la classe d'un individu.

Mettons que nous avons accès à une feature d'intérêt pour cette classification. 
On notera $X$ l'espace des **observations** associé.

Pour déterminer la classe d'un individu, on peut alors partir du principe suivant : choisir le $C_i$ tel que $p(C_i \mid x)$ **soit maximal**.

On nomme $p(C_i \mid x)$ "**probabilités a posteriori**".

D'après le **théorème de Bayes** :

$p(C_i \mid x) = \frac{p(x \mid C_i)p(C_i)}{p(x)}$

avec $p(x) = \sum{i=1}{q} p(x \mid C_i)p(C_i)$

On nomme $p(x)$ la densité de "**probabilité d'observation**", et $p(x \mid C_i)$ la densité de "**probabilité conditionnelle d'observation**".

Toute la difficulté de la méthode est d'**estimer** $p(x \mid C_i)$.
On va en général chercher à **modéliser** ces densités de probabilité conditionnelle.

![Décision Bayesienne](img/Chap2_decision_bayesienne.png)

**Attention !** En général, il y a des recouvrements entre les différentes densités de probabilité conditionnelle.
On ne peut alors pas obtenir classifieur parfait.
On cherchera juste le modèle permettant de minimiser les erreurs de classification. 

**Cas particulier :** Si tous les $p(x \mid C_i)$ sont égaux, alors la feature sélectionnée n'est pas pertinente pour la classification.

Ce principe est **généralisable** aux cas de classifications avec $m$ features d'espaces de probabilité $X_1$, $X_2$, ... $X_m$.
On cherchera la classe $C_i$ qui maximise $p(C_i) \prod{j=1}{m}p(x_j \mid C_i)$.

#### Choix du modèle

Comme nous l'avons expliqué, la décision Bayesienne nécessite un modèle des probabilités conditionnelles d'observation $p(x \mid C_i)$ pour chaque classe $C_i$.

Pour ce faire, on **ajuste une fonction de densité de probabilité** pour chaque $p(C_i \mid x)$ à notre jeu d'entrainement.

Ceci implique donc 2 choix :

- Une **fonction de densité de probabilité**, ce qui implique de faire une hypothèse sur la distribution des observations pour chaque classe.

- Une **méthode d'ajustement de loi de probabilité**.

Parmi les fonctions de densité de probabilité classiques, on peut citer : la loi normale, la loi uniforme ou la loi de Student.

La méthode d'ajustement la plus classique pour la décision Bayesienne est celle du **maximum de vraisemblance**.

#### Maximum de vraisemblance

#### Implémentation Scikit-Learn

#### Application à notre exemple

![Histogramme de la 1ère harmonique pour la flute et la trompette](img/Chap2_exemple_histogramme.png)

![Probabilités conditionnelles](img/Chap2_exemple_probabilites_conditionnelles.png)

![Probabilité d'observation](img/Chap2_exemple_probabilite_observation.png)

![Probabilités a posteriori](img/Chap2_exemple_probabilites_aposteriori.png)

### K Plus Proches Voisins

#### Principe

#### Choix de la distance

#### Choix du paramètre K

#### Implémentation Scikit-Learn

#### Application à notre exemple

### Perceptron

#### Neurone artificiel

#### Apprentissage et descente de gradient

#### Réseau de neurones

#### Implémentation Scikit-Learn

#### Application à notre exemple
