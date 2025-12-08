# Chapitre II : Classification supervisée

Ce chapitre est une introduction à la classification supervisée : principe, mesures de performances et méthodes de base.

![En-tête chapitre II](img/Chap2_header.png)

---

## Problème de classification supervisée

Comme mentionné lors du Chapitre I, par "**classifier**" on entend associer une réalisation d'une variable **quantitative discrète** ou **qualitative** à un individu (labels), à partir des réalisations d'autres variables (features).
On appelle ces labels des "**classes**".

On parlera ici de "classification supervisée" car on va entrainer un modèle (aussi appelé "classifieur") à associer une classe à des individus, en se basant sur des données déjà labélisées.
Il s'agit donc bien d'un **apprentissage supervisé**.

L'idée est que le classifieur soit ensuite capable de **généraliser** : prédire la "classe" d'un nouvel individu.

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
import pandas as pd
df_dataset = pd.read_csv(input_path)
~~~

Il est possible avec Seaborn d'afficher ces données sous la forme d'une **matrice de corrélations**, avec chaque classe d'une couleur différente.
Ce type de représentation permet de vérifier la séparabilité des différentes classes à partir des features sélectionnés.

Voici la commande Seaborn :

~~~
import seaborn as sns
sns.pairplot(df_dataset,hue='instrument')
~~~

On obtient alors le graphique suivant :

![Matrice de corrélations des 3 instruments](img/Chap2_correlation_matrix_instruments.png)

On observe que les classes "flute", "oboe" et "trumpet" sont plutôt bien séparables à partir des amplitudes des 3 premières harmoniques.
Vouloir entrainer un modèle à reconnaitre un de ces instruments à partir de ces données à donc du sens.

**Il est à noter que nous avons ici grandement simplifié le problème et sa résolution pour les besoins de ce cours.**
**Nous verrons cet exemple plus en détails en TP.**

## Mesures de performance

Nous allons passer en revue dans cette section les principaux indicateurs de performances applicables à tous les types de classification.

### Matrice de confusion

Pour chaque classe $C$ possible, lorsqu'un classifieur réalise une prédiction sur un individu, il y a 4 possibilités :

* Le classifieur a prédit $C$, et l'individu appartient bien à $C$ : c'est un **vrai positif** (noté TP).

* Le classifieur a prédit $C$, et l'individu n'appartient pas à $C$ : c'est un **faux positif** (noté FP).

* Le classifieur n'a pas prédit $C$, et l'individu n'appartient pas à $C$ : c'est un **vrai négatif** (noté TN).

* Le classifieur n'a pas prédit $C$, et l'individu appartient bien à $C$ : c'est un **faux négatif** (noté FN).

Tous les scores de performance pour la classification que nous allons voir se basent sur le nombre de TP, FP, TN, FN obtenus par le modèle sur un jeu d'individus labélisé.

Les indicateurs brutes que sont le nombre de TP, FP, TN et FN sont en général mis sous la forme d'un tableau, que l'on appelle **matrice de confusion**.



### Exactitude et précision

### Précision-rappel et score F1

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
Nous expliquerons pourquoi dans la suite.

On nomme $p(C_i \mid x)$ "**probabilités a posteriori**".

D'après le **théorème de Bayes** :

$p(C_i \mid x) = \frac{p(x \mid C_i)p(C_i)}{p(x)}$

avec $p(x) = \sum_{i=1}^{q} p(x \mid C_i)p(C_i)$

On nomme $p(x)$ la densité de "**probabilité d'observation**", et $p(x \mid C_i)$ la densité de "**probabilité conditionnelle d'observation**".

Toute la difficulté de la méthode est d'**estimer** $p(x \mid C_i)$.
On va en général chercher à **modéliser** ces densités de probabilité conditionnelle.

![Décision Bayesienne](img/Chap2_decision_bayesienne.png)

**NB :** Il est à noter que rechercher la classe $C_i$ maximisant $p(C_i \mid x)$ **est équivalent** à rechercher $C_i$ maximisant $p(x \mid C_i)p(C_i)$.
Il n'est donc en théorie pas utile de calculer $p(x)$ pour obtenir le classifieur.
Mais il est nécessaire d'avoir $p(x)$ pour obtenir des probabilités d'appartenance à une classe.

**Attention !** En général, il y a des recouvrements entre les différentes densités de probabilité conditionnelle.
On ne peut alors pas obtenir classifieur parfait.
On cherchera juste le modèle permettant de minimiser les erreurs de classification. 

**Cas particulier :** Si tous les $p(x \mid C_i)$ sont égaux, alors la feature sélectionnée n'est pas pertinente pour la classification.

Ce principe est **généralisable** aux cas de classifications avec $m$ features d'espaces de probabilité $X_1$, $X_2$, ... $X_m$.
On cherchera la classe $C_i$ qui maximise $p(C_i) \prod_{j=1}^{m}p(x_j \mid C_i)$.

#### Frontière de décision et erreur

Comme nous l'avons expliqué précédemment, sauf cas particulier, on ne peut pas obtenir un classifieur parfait.

On va donc essayer d'établir des **frontières de décision** entre les classes : des intervalles de $x$ pour lesquels on attribura une classe.
Et nous recherchons même les frontières de décision optimales : celles qui minimisent le risque d'erreurs.

Prenons un cas simple de classification binaire entre 2 classes $C_1$ et $C_2$.
Nous noterons $x = D$ la frontière de décision choisie.

On a alors 2 types d'erreurs de classification possibles : 

* Classifier l'individu en $C_1$ alors qu'il appartient à $C_2$.

* Classifier l'individu en $C_2$ alors qu'il appartient à $C_1$.

Les probabilités d'erreurs associées sont $\int_{-\infty}^{D} p(x \mid C_2)p(C_2) dx$ et $\int_{D}^{+\infty} p(x \mid C_1)p(C_1) dx$.

Elles correspondent aux aires représentées sur ce schéma : 

![Erreur décision Bayesienne](img/Chap2_erreur_decision_bayesienne.png)

Pour obtenir la frontière de décision optimale $x = O$, on va chercher à minimiser la somme de ces erreurs.

L'aire entourée en vert correspond à ce que l'on appelle "l'**erreur réductible**" : c'est la portion de l'erreur totale que l'on peut réduire pour obtenir la frontière optimale.

On retrouve bien que :

* Si $p(C_1 \mid x) > p(C_2 \mid x)$ alors on classifie l'individu comme appartenant à $C_1$.

* Si $p(C_1 \mid x) < p(C_2 \mid x)$ alors on classifie l'individu comme appartenant à $C_2$.

On peut généraliser à $q$ classes : comme dit précédemment, pour obtenir les frontières de décision optimales, on choisi le $C_i$ tel que $p(C_i \mid x)$ **soit maximal**.

#### Choix du modèle

Comme nous l'avons expliqué, la décision Bayesienne nécessite un modèle des probabilités conditionnelles d'observation $p(x \mid C_i)$ pour chaque classe $C_i$.

Pour ce faire, on **ajuste une fonction de densité de probabilité** pour chaque $p(C_i \mid x)$ à notre jeu d'entrainement.

Ceci implique donc 2 choix :

- Une **fonction de densité de probabilité**, ce qui implique de faire une **hypothèse forte** sur la distribution des observations pour chaque classe.

- Une **méthode d'ajustement de loi de probabilité**.

La fonction de densité de probabilité la plus classique est celle de la **loi normale** : 

$f(x) = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{1}{2} (\frac{x - \mu}{\sigma})^2}$

avec 2 paramètres à ajuster $\mu$ (la moyenne) et $\sigma$ (l'écart-type).

La méthode d'ajustement la plus classique pour la décision Bayesienne est celle du **maximum de vraisemblance**.
C'est elle que nous allons détailler.

#### Maximum de vraisemblance

|Définition|
|:-|
|Soit une loi de probabilité $f(x,\theta)$, définie par des paramètres $\theta$.|
|Pour un échantillon observé $(x_1,x_2,...,x_n)$, on nomme **vraisemblance** ("likelihood" en anglais) la probabilité que cet échantillon provienne d'un tirage de $f(x,\theta)$.|
|Si les tirages sont indépendants, on peut exprimer la vraisemblance de la manière suivantes :|
|$L(x_1,x_2,...,x_n,\theta) = \prod_{k=1}^{n} f(x_k,\theta)$|

La méthode du **maximum de vraisemblance** découle du fait que le modèle $f$ de paramètres $\theta$ représentant le mieux les observations est celui qui **maximise** la vraisemblance, c'est-à-dire **la probabilité que l'échantillon provienne de cette loi**.

L'idée est donc de rechercher les $\theta$ maximisant $L(x_1,x_2,...,x_n,\theta)$.

Souvent, pour simplifier les calculs, on ne va pas rechercher le maximum de la vraisemblance, mais de la log-vraisemblance :

$logL(x_1,x_2,...,x_n,\theta) = \sum_{k=1}^{n} log(f(x_k,\theta))$

En effet, rechercher les paramètres $\theta$ maximisant $L$ ou $logL$ est équivalent, et rechercher un maximum implique un calcul de dérivée, ce qui est plus simple pour des sommes que pour des produits.

On va donc pour chaque paramètre $\theta_j$ de $\theta$ la valeur qui vérifie $\frac{\partial}{\partial{\theta_j}} \sum_{k=1}^{n} log(f(x_k,\theta)) = 0$.

Prenons l'exemple de la loi normale :

$f(x,\mu,\sigma) = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{1}{2} (\frac{x - \mu}{\sigma})^2}$

On cherchera alors les paramètres $\theta = (\mu,\sigma)$ vérifiant :

$\frac{\partial}{\partial{\theta}} \sum_{k=1}^{n} (-log(\sigma) - log(\sqrt{2 \pi}) - \frac{1}{2} (\frac{x - \mu}{\sigma})^2) = 0$

soit $\frac{\partial}{\partial{\theta}} (- n log(\sigma) - n log(\sqrt{2 \pi}) - \sum_{k=1}^{n} \frac{1}{2} (\frac{x - \mu}{\sigma})^2) = 0$

soit pour chaque paramètre :

$\frac{\partial}{\partial{\mu}} (- n log(\sigma) - n log(\sqrt{2 \pi}) - \sum_{k=1}^{n} \frac{1}{2} (\frac{x - \mu}{\sigma})^2) = 0$

$\frac{\partial}{\partial{\sigma}} (- n log(\sigma) - n log(\sqrt{2 \pi}) - \sum_{k=1}^{n} \frac{1}{2} (\frac{x - \mu}{\sigma})^2) = 0$

On montre alors que les paramètres vérifiant ces équations sont :

$\mu = \frac{1}{n} \sum_{k=1}^{n} x_k$ et $\sigma^2 = \frac{1}{n} \sum_{k=1}^{n} (x_k - \mu)^2$

Ce qui était attendu.

#### Implémentation Scipy

Afin de réaliser un ajustement de loi de probabilité, on peut utiliser la bibliothèque de calculs scientifiques Scipy, et en particulier son module de statistiques "scipy.stat".

Par exemple, pour un ajustement avec une loi normale, on pourra importer l'objet "norm" avec :

~~~
from scipy.stats import norm
~~~

On peut alors ajuster une loi normale à un ensemble d'observations contenu dans un conteneur `x_obs`, et récupérer la moyenne `mu` et l'écart-type `sigma` avec :

~~~
mu,sigma = norm.fit(x_obs)
~~~

Par défaut, la méthode du maximum de vraisemblance est utilisée.
Mais on peut également utiliser la "méthode des moments" (que nous ne présenterons pas dans ce cours), en ajoutant un paramètre `method = 'MM'` en entrée.

Une fois la loi normale ajustée, on a accès à la densité de probabilité `dp` associée à une réalisation `x` avec :

~~~
dp = norm.pdf(x,mu,sigma)
~~~

Bien d'autres lois de probabilité sont disponibles dans le module "scipy.stats", et fonctionnent sur le même principe que "norm".

#### Implémentation Scikit-Learn

Il est à noter qu'il existe aussi une implémentation de la classification Bayesienne dans Scikit-Learn, dans l'hypothèse de distributions des features suivant des lois normales.

Elle peut être importée avec :

~~~
from sklearn.naive_bayes import GaussianNB
~~~

On doit alors créer un objet "GaussianNB" qui contiendra notre modèle :

~~~
bayes_classifier = GaussianNB()
~~~

Il faut ensuite l'entrainer avec nos features et labels d'entrainement, nommés ici `feature_train` et `label_train` :

~~~
bayes_classifier.fit(feature_train,label_train)
~~~

Et enfin, on peut réaliser une prédiction à partir de features de test, nommés `feature_test` :

~~~
label_test = bayes_classifier.predict(feature_test)
~~~

Cette implémentation peut être pratique dans certains cas, mais elle ne permet pas de jouer sur les hyperparamètres suivants : la loi de probabilité et la méthode d'ajustement.
Une optimisation de ces hyperparamètres n'est donc pas possible avec Scikit-Learn.

#### Application à notre exemple

Nous allons à présent appliquer la classification Bayesienne à notre problème exemple.

Afin de rendre la visualisation plus facile, nous allons simplifier le problème :

Mettons que nous voulons juste effectuer une classification binaire de nos enregistrements, entre les classes "flute" ou "trompette", en utilisant comme unique feature l'amplitude relative de la 1ère harmonique.

Pour ce faire, nous importons le fichier CSV depuis son chemin `input_path` sous la forme d'un DataFrame, et nous sélectionnons les variables et les individus qui nous intéressent :

~~~
df_dataset = pd.read_csv(input_path)

df_dataset = df_dataset[['instrument','harmo1']]
df_dataset = df_dataset[(df_dataset['instrument']=='flute')|(df_dataset['instrument']=='trumpet')]
~~~

Nous diviserons ici nos données en un jeu d'entrainement (80%) et un jeu de test (20%), sous la forme de 2 DataFrames : 

~~~
df_train=df_dataset.sample(frac=0.8,random_state=0)
df_test=df_dataset.drop(df_train.index)
~~~

On peut alors tracer un histogramme de notre feature pour les données d'entrainement, en sépararant les 2 classes :

![Histogramme de la 1ère harmonique pour la flute et la trompette](img/Chap2_exemple_histogramme.png)

On peut noter qu'il y a peu de recouvrement entre les 2 distributions, ce qui laisse entrevoir qu'il est possible d'entrainer un modèle à classifier ces données.

En 1ère approche, nous choisissons d'ajuster à ces 2 distributions des modèles de lois normales.
Il faudra se poser la question de la pertinence de ce choix.

Tout d'abord, nous allons séparer le jeu d'entrainement en 2 Series (DataFrame Pandas ne contenant qu'une colonnes) suivant si l'instrument est une flute ou une trompette :

~~~
sr_harmo1_flute_train = df_train[df_train['instrument']=='flute']['harmo1']
sr_harmo1_trumpet_train = df_train[df_train['instrument']=='trumpet']['harmo1']
~~~

On peut alors réaliser nos 2 ajustements, et récupérer les paramètres $mu$ et $\sigma$ correspondants :

~~~
from scipy.stats import norm

mu_harmo1_flute,sig_harmo1_flute = norm.fit(sr_harmo1_flute_train)
mu_harmo1_trumpet,sig_harmo1_trumpet = norm.fit(sr_harmo1_trumpet_train)
~~~

Maintenant que nous avons les paramètres de nos modèles, nous pouvons évaluer la densité des probabilités conditionnelles pour la classe "flute" et la classe "trompette".

Voici par exemple 301 évaluations pour des valeurs d'amplitude de la 1ère harmonique entre -30 et 0 dB :

~~~
x_axis = np.linspace(-30,0,301)

proba_norm_flute = norm.pdf(x_axis,mean_harmo1_flute,sig_harmo1_flute)
proba_norm_trumpet = norm.pdf(x_axis,mean_harmo1_trumpet,sig_harmo1_trumpet)
~~~

Nous pouvons alors tracer les courbes correspondantes par-dessus notre histogramme (affiché en densité de probabilité) :

![Probabilités conditionnelles](img/Chap2_exemple_probabilites_conditionnelles.png)

Si nos modèles ne paraissent pas complément inadaptés, on peut noter qu'ils ne capturent pas la légère asymétrie de nos distributions.
On pourrait donc se poser la question d'essayer d'autres lois de probabilités, asymétriques.

Continuons avec nos modèles pour les probabilités conditionnelles.

La prochaine étape est d'estimer la probabilité de chaque classe, à partir de leurs densités relatives :

~~~
proba_flute = len(sr_harmo1_flute_train)/len(df_train)
proba_trumpet = len(sr_harmo1_trumpet_train)/len(df_train)
~~~

On peut alors utiliser estimer la densité de probabilité d'observation :

~~~
proba_obs = proba_norm_flute*proba_flute + proba_norm_trumpet*proba_trumpet
~~~

Si nous l'affichons avec les histogrammes, nous pouvons vérifier qu'elle est bien cohérente avec la distribution des observations.

![Probabilité d'observation](img/Chap2_exemple_probabilite_observation.png)

Enfin, nous pouvons calculer les probabilités a posteriori, en se basant sur la formule de Bayes :

~~~
proba_bayes_flute = proba_norm_flute*proba_flute/proba_obs
proba_bayes_trumpet = proba_norm_trumpet*proba_trumpet/proba_obs
~~~

On peut alors afficher ces probabilités, et tracer la frontière de décision :

![Probabilités a posteriori](img/Chap2_exemple_probabilites_aposteriori.png)

La frontière de décision se trouve à environ -13.16 dB : 

* Si on mesure une 1ère harmonique ayant une amplitude inférieure, on classifiera l'instrument comme étant une flute.

* Si on mesure une 1ère harmonique ayant une amplitude supérieure, on classifiera l'instrument comme étant une trompette.

Comme nous l'avons mentionné précédemment, si la probabilité d'appartenance aux classes ne nous intéresse pas, nous pourrions juste comparer $p(x \mid C = 'flute')p(C='flute')$ et $p(x \mid C = 'trumpet')p(C='trumpet')$ pour classifier les observations. 

Maintenant que nous avons vu le principe, on voudrait pouvoir ré-entrainer notre modèle afin d'optimiser les hyperparamètres, et réaliser des prédictions sur les jeux d'entrainement et de test, le tout de manière efficace.

Dans ce but, nous pouvons mettre notre classification binaire Bayesienne sous la forme d'une classe `binary_bayes` avec 2 méthodes `train` et `predict` pour l'entrainement et la prédiction.
Voici un exemple d'implémentation :

~~~
class binary_bayes:
    
    def __init__(self,stat_model):
        
        self.stat_model = stat_model
        
        self.true_params = None
        self.false_params = None
        
        self.proba_true = None
    
    def train(self,x_true,x_false):
        
        self.true_params = self.stat_model.fit(x_true)
        self.false_params = self.stat_model.fit(x_false)
        
        len_true = len(x_true)
        len_false = len(x_false)
        self.proba_true = len_true/(len_true+len_false)
        
    def predict(self,x):
        
        proba_norm_true = self.stat_model.pdf(x,*self.true_params)
        proba_norm_false = self.stat_model.pdf(x,*self.false_params)
        
        proba_obs = proba_norm_true*self.proba_true + proba_norm_false*(1-self.proba_true)

        proba_bayes_true = proba_norm_true*self.proba_true/proba_obs
        
        return proba_bayes_true
~~~

On peut alors facilement définir un classifieur binaire "est-ce une flute ?" utilisant la loi normale telle qu'implémentée par Scipy :

~~~
from scipy.stats import norm

is_a_flute = binary_bayes(norm)
~~~

Entrainer ce classifieur sur notre jeu d'entrainement :

~~~
is_a_flute.train(sr_harmo1_flute_train,sr_harmo1_trumpet_train)
~~~

Et réaliser des prédictions sur nos données d'entrainement et de test :

~~~
prediction_train = (is_a_flute.predict(df_train['harmo1']))

prediction_test = (is_a_flute.predict(df_test['harmo1']))
~~~

En partant du principe que nous positionnons la frontière de décision à une probabilité d'appartenance à classe "flute" de 0.5, nous pouvons obtenir les matrices de confusion en entrainement et en test avec les commandes suivantes :

~~~
from sklearn.metrics import confusion_matrix

#Label encoding:
ground_truth_train = (df_train['instrument']=='flute').astype(int)
ground_truth_test = (df_test['instrument']=='flute').astype(int)

cm_train = confusion_matrix(ground_truth_train, prediction_train>0.5)
cm_test = confusion_matrix(ground_truth_test, prediction_test>0.5)
~~~

Voici les résultats en entrainement obtenus pour notre exemple :

![Exemple de matrice de confusion](img/Chap2_exemple_matrice_confusion.png)

On observe que les performances du modèle sont très similaires entre les données d'entrainement et de test.
Ceci tend à montrer que l'on a pas de problème de sur-ajustement important, ce qui laisse présager des performances similaires en généralisation.

Il n'y a aucun faux positif, mais on a quelques faux négatifs : parfois notre modèle classifie des enregistrements de flutes comme n'étant pas des flutes.

Suivant les applications, on peut vouloir choisir une frontière de décision différente, pour diminuer le nombre de faux négatifs, au prix d'une augmentation du nombre de faux positifs.
Afin de voir les effets d'un tel choix, on tracer une courbe ROC à partir des probabilités prédites par notre modèle :

~~~
fp_rate_train, tp_rate_train, thresholds_train = roc_curve(ground_truth_train, prediction_train)
fp_rate_test, tp_rate_test, thresholds_test = roc_curve(ground_truth_test, prediction_test)
~~~

#### Remarques

La méthode de la classification de Bayes a les **avantages** suivants :

* Elle fonctionne pour **tous les types de classification et variables**.

* Elle est relativement **simple** à mettre en place, avec **peu de paramètres**.

* Les décisions qu'elle prend sont complément **expliquées** et **interprétables** : un humain peut les comprendre.

Mais cette méthode a aussi les **limites** suivantes :

* Elle fait l'hypothèse de l'**indépendance des variables** entre elles, ce qui dans la pratique limite son application aux problèmes avec peu de features.

* Elle fait une hypothèse forte sur la **distribution des observations** pour chaque variable. Il s'agit souvent d'une hypothèse de **normalité**.

### K Plus Proches Voisins

#### Principe

La méthode de la classification Bayesienne que nous venons de voir avait pour désavantage de nécessiter une hypothèse sur la distribution des observations.

Dans cette section, nous allons présenter une méthode ne nécessitant aucun a priori sur les données : les **K Plus Proches Voisin**, aussi connue sous l'acronyme KPPV.

Les KPPV est une méthode dite de "lazy learning" : il n'y a pas de réel apprentissage préalable à la prédiction.
Le jeu de données d'apprentissage est **stocké en mémoire**, et utilisé au moment de la prédiction.

L'idée est la suivante : pour classer un nouvel individu, on va calculer sa **distance aux $k$ individus les plus proches** dans les données d'entrainement.
On attibura alors à l'individu la classe **la plus représentée** parmi ses $k$ "plus proches voisins".

![K plus proches voisins](img/Chap2_kppv.png)

Prédire la classe d'un individu avec cette méthode implique :

(1) De mesurer la distance entre l'individu à classifier et **tous les individus du jeu d'entrainement**.
C'est ce que l'on appelle "l'approche brute". 
Et plus le jeu d'entrainement est grand, plus le temps de calcul sera long.
Pour cette raison, on choisi de stocker le jeu d'entrainement dans une **structure de donnée la plus efficace à parcourir** possible (exemple : KD-Tree).

(2) De choisir une **mesure de distance** adaptée au problème.

(3) De choisir le **nombre de "plus proches voisins"** à l'individu à considérer.

(4) De choisir de quelle manière on va affecter une classe à l'individu à partir de la classe de ses voisins : Un **vote majoritaire** ? 
S'il y a un gros déséquilibre entre classes, ce type de vote risque d'être biaisé.
On préférera alors un vote avec des **poids différents** suivant les classes.

#### Choix de la distance

Suivant le problème de classification auquel on est confronté, la "distance" entre 2 individus n'a pas le même sens.

En effet, on comprend bien qu'on utilisera pas les mêmes critères pour mesurer la distance entre 2 valeurs réelles, entre 2 images, ou entre 2 mots du dictionnaire.

D'où l'importance lorsqu'on utilise les KPPV de **choisir une mesure de distance pertinente** pour notre problème.

Parmi les mesures de distances classiques, on peut citer :

* **Distance Euclidienne** : 

Si on veut mesurer la distance Euclidienne entre $x$ et $y$, 2 vecteurs de dimension $n$ :

$D(x,y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$

Il s'agit de la mesure de distance la plus connue et la plus utilisée.
Elle fonctionne bien lorsque l'on est confrontés à des valeurs réelles continues, normalisées, et avec une dimensionnalité faible.

Cette distance peut être vue comme la mesure de distance associée à la norme 2.

* **Distance de Manhattan** :

Si on veut mesurer la distance de Manhattan entre $x$ et $y$, 2 vecteurs de dimension $n$ :

$D(x,y) = \sum_{i=1}^{n} \mid x_i - y_i \mid$

Suivant l'espace des features de notre problème, tracer une "ligne droite" entre individus peut ne pas avoir de sens.
La distance de Manhattan est alors une alternative à la distance Euclidienne.

Cette distance peut être vue comme la mesure de distance associée à la norme 1.

* **Distance de Chebychev** :

Si on veut mesurer la distance de Chebychev entre $x$ et $y$, 2 vecteurs de dimension $n$ :

$D(x,y) = max(\mid x_i - y_i \mid)$

La distance de Chebychev est assez peu utilisée, car elle a des cas d'applications très spécifiques.
(Par exemple, les déplacements d'un roi sur un jeu d'échec ou les automates cellulaires).

Cette distance peut être vue comme la mesure de distance associée à la norme infinie.

* **Minkowski** :

Si on veut mesurer la distance de Minkowski entre $x$ et $y$, 2 vecteurs de dimension $n$ :

$D(x,y) = (\sum_{i=1}^{n} \mid x_i - y_i \mid^p)^{1/p}$

La distance de Minkowski est une généralisation des 3 distances précédentes.
En effet, on remarque que si $p=1$ elle revient à la distance de Manhattan, si $p=2$ elle revient à la distance Euclidienne, et si $p$ tend vers l'infini elle revient à la distance de Chebychev.

Elle permet donc de chercher un compromis entre ces différentes distances.

* **Hamming** :

Soit 2 chaines de caractères de même taille. 
La distance de Hamming entre ces 2 chaines est alors égale au nombre de positions pour lesquelles les caractères sont différents.

Cette mesure de distance est couramment utilisée lorsque l'on veut comparer des morceaux de textes caractère par caractère, ou de manière générale pour des données qualitatives.

* **Similarité cosinus** :

Si on veut mesurer la "similarité cosinus" entre 2 vecteurs $x$ et $y$ :

$D(x,y) = cos(\theta) = \frac{x.y}{\|x\| \|y\|}$

Il s'agit du cosinus de l'angle entre les 2 vecteurs.

Cette mesure de distance est couramment utilisée lorsque l'on doit comparer des vecteurs de haute dimensionnalité, et où la norme du vecteur a peu d'importance.
Par exemple, c'est la mesure de distance privilégiée pour de la "fouille de texte" (comparaison mot à mot de chaines de caractère).

![Exemples de distances](img/Chap2_distances.png)

La distance est donc un **hyperparamètre à optimiser** lorsque l'on utilise les KPPV.

#### Choix du paramètre K

Il est évident que le choix de $k$ va avoir un impact sur les prédictions obtenues à partir des données d'entrainement.
C'est donc également un **hyperparamètre à optimiser**.

Pour choisir des valeurs de $k$ à tester, on peut partir des grands principes suivants :

* S'il y a un fort **déséquilibre** entre classes, il vaut mieux ne pas choisir un $k$ **faible**.

* S'il y a beaucoup de **recouvrement** entre les classes, il vaut mieux choisir un $k$ **élevé**.

* Avec un $k$ trop **faible** on risque le **sur-apprentissage**.

* Avec un $k$ trop **grand** on risque le **sous-apprentissage**.

Pour éviter les cas d'égalité, on va en général choisir une valeur de $k$ impaire.

#### Implémentation Scikit-Learn

Il existe une implémentation Scikit-Learn de la méthode des KPPV.

Elle peut être importée avec :

~~~
from sklearn.neighbors import KNeighborsClassifier
~~~

On peut ensuite initialiser un classifieur KPPV avec un objet "KNeighborsClassifier" de paramètre `k` correspondant au nombre de plus proches voisins :

~~~
knn = KNeighborsClassifier(n_neighbors=k)
~~~

Pour donner le jeu d'entrainement (features avec `feature_train` et labels avec `label_train`) à ce classifieur, on utilise la méthode :

~~~
knn.fit(feature_train,label_train)
~~~

On peut à présent réaliser des prédictions `label_test` à partir de features de test `feature_test` :

~~~
label_test = knn.predict(feature_test)
~~~

Si on veut effectuer un test de notre classifieur sur un jeu de données labéliser, on peut obtenir un score d'exactitude avec la commande :

~~~
knn.score(feature_test,label_test)
~~~

#### Outils de visualisation MLxtend

Pour afficher les frontières de décision données par un classifieur dans un cas 1D ou 2D, il existe une fonction de la bibliothèque "MLxtend".

Une fois la bibliothèque installée, vous pouvez importer la fonction avec :

~~~
from mlxtend.plotting import plot_decision_regions
~~~

Pour afficher les frontières de décision d'une classifieur `model`, avec les données d'entrainement `feature_train` et `label_train`, on utilisera la méthode :

~~~
plot_decision_regions(feature_train, label_train, clf=model)
~~~

Pour un problème de dimensionnalité plus élevée que 2, la visualisation des frontières de décision est toujours difficile.

#### Application à notre exemple

Nous allons à présent appliquer les KPPV à notre problème exemple.

Afin de rendre la visualisation plus facile, nous allons simplifier le problème :

Mettons que nous voulons effectuer une classification de nos enregistrements entre les classes "flute", "hautbois" ou "trompette", en utilisant en features l'amplitude relative de la 1ère harmonique et de la 2ème harmonique.

Tout d'abord, nous importons notre fichier CSV sous la forme d'un DataFrame, depuis le chemin `input_path` :
~~~
df_dataset = pd.read_csv(input_path)
~~~

Même si en théorie les KPPV n'ont pas besoin de labels numériques pour fonctionner, certaines des fonctions que nous utiliserons dans la suite ne fonctionnent qu'avec des valeurs numériques.
Nous allons donc encoder les labels "par étiquette" :

~~~
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_dataset['instrument'] = encoder.fit_transform(df_dataset['instrument'])
~~~

Nous récupérons ensuite les features et les labels que nous allons utiliser dans 2 DataFrames :

~~~
df_features = df_dataset[['harmo1','harmo2']]
df_labels = df_dataset['instrument']
~~~

Nous séparons ensuite nos données en un jeu d'entrainement (80%) et un jeu de test (20%), sous la forme de 4 DataFrames (2 pour les features, 2 pour les labels) : 

~~~
from sklearn.model_selection import train_test_split
df_features_train, df_features_test, df_labels_train, df_labels_test = train_test_split(df_features,df_labels,test_size=0.2,random_state=0)
~~~

Maintenant que les données sont prêtes, nous pouvons créer notre classifieur.
Voici comment initialiser un classifieur KPPV avec $k=3$ :

~~~
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
~~~

Pour lui fournir les données d'entrainement, comme vu précédemment, il nous suffit d'utiliser la commande suivante :

~~~
knn.fit(df_features_train,df_labels_train)
~~~

Nous pouvons à présent utiliser notre modèle pour classifier des données.
Tout d'abord, nous allons évaluer les performances de notre modèle en entrainement et en test.

On peut déjà mesurer l'exactitude de notre classifieur sur ces 2 jeux de données :

~~~
print(knn.score(df_features_train,df_labels_train))
print(knn.score(df_features_test,df_labels_test))
~~~

Pour $k=3$, on obtient environ 99.4% d'exactitude en entrainement, et environ 98.8% d'exactitude en test.
Ces scores laissent à penser que notre modèle aura de plutôt bonnes performances en généralisation, mais il nous faut une matrice de confusion complète pour conclure :

![Exemple de matrice de confusion](img/Chap2_exemple_matrice_confusion_2.png)

On observe qu'à l'entrainement comme en test, le hautbois est très bien séparé des autres instruments, alors que la trompette et la flute sont parfois confondus.
Ce résultat était prévisible au vu de la matrice de corrélations que nous avions obtenue lors de notre étude préliminaire.
Les proportions d'erreurs restent cependant relativement faibles comparées aux nombres d'observations.

Nous n'avons pour l'instant testé qu'une valeur de $k$.
Pour visualiser l'effet de cet hyperparamètre, nous pouvons utiliser les affichages graphiques de la bibliothèque MLxtend.

Voici les graphiques obtenus pour $k=3$ (volontairement faible) et $k=31$ (volontairement élevé) :

![Frontières de décision pour notre exemple](img/Chap2_exemple_KPPV_frontieres_decision.png)

On peut noter que comme attendu, le choix de $k$ a le plus d'effet à la frontière entre "flute" et "trompette".
En effet, comme il y a du recouvrement entre ces 2 classes, on sait qu'il vaut mieux choisir un $k$ élevé pour éviter le sur-apprentissage.
Ceci est confirmé par le "lissage" de la frontière de décision lorsque l'on utilise $k=31$.

Le choix d'un $k$ élevé a donc l'air plus approprié ici, mais il faudrait réaliser une réelle optimisation de cet hyperparamètre.

Par défaut, la distance utilisée par l'implémentation Scikit-Learn des KPPV est la distance Euclidienne.
Nous pouvons également réaliser des affichages pour visualiser l'impact de différentes distances pour une même valeur de $k$.

Voici le résultat pour la distance Euclidienne et la distance de Manhattan, avec $k=31$.

![Frontières de décision pour notre exemple](img/Chap2_exemple_KPPV_frontieres_decision_2.png)

On observe en effet que le choix de la distance impacte significativement les frontières de décision obtenues, même s'il est difficile ici de juger de la pertinence d'une des 2 distances essayées.
Tout comme pour $k$, il faudrait réaliser une véritable optimisation de cet hyperparamètre.

#### Remarques

La méthode des KPPV a les **avantages** suivants :

* Il s'agit d'une méthode non-paramétrique, qui ne fait **aucune hypothèse** sur la structure des données.

* Elle est relativement simple, et n'a **que 2 hyperparamètres** ($k$ et la distance), ce qui est peu comparé à certaines méthodes.

* Si de nouvelles observations doivent être ajoutées au jeu d'entrainement, **la mise à jour du modèle est directe**.

Mais cette méthode a aussi les **limites** suivantes :

* Le modèle ayant besoin de stocker les données d'entrainement, il peut vite devenir **très lourd**.

* Elle fonctionne mal avec des données de **grande dimension**.

* Elle est très sujette au **sur-apprentissage**.

### Perceptron

#### Neurone artificiel

#### Apprentissage et descente de gradient

#### Réseau de neurones

#### Implémentation Scikit-Learn

#### Application à notre exemple
