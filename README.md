# Prédiction des retards de vols

Système qui prédit la classe du retard à l'arrivée d'un vol :
- classe 0 : à l'heure (retard ≤ 15 minutes) ;
- classe 1 : retard entre 15 et 30 minutes ;
- classe 2 : retard supérieur à 30 minutes.

> Projet académique d'équipe (6 personnes, 3 binômes), réalisé en 2026.
> **Équipe :** Khaoula El Kihel, Hajar Harbouli, Assiya Ouaradi,
> Meryem El Hajjoui, Hajar El Maimouni et Nassira Ouzzine.
> **Ma contribution (Nassira Ouzzine) :** partie machine learning, en binôme
> avec Meryem El Hajjoui : entraînement et comparaison de la régression
> logistique, du Random Forest, du SVM et de XGBoost, réglage de XGBoost par
> GridSearchCV, évaluation (matrices de confusion, ROC-AUC, courbes ROC et
> Precision-Recall) et choix du meilleur modèle.

## Organisation

Le travail est réparti en trois binômes, reliés par un script final :
1. prétraitement des données ;
2. machine learning : régression logistique, forêt aléatoire, SVM, XGBoost ;
3. deep learning : MLP (Optuna), LSTM, GRU, Transformer (PyTorch).

## Contenu du dépôt

- `Notebook_ML.ipynb` : exploration, nettoyage, ingénierie des variables,
  encodage et modèles de machine learning.
- `Notebook_DL.ipynb` : modèles de deep learning.
- `pipeline_automatique_final1.py` : script qui enchaîne prétraitement,
  machine learning et comparaison avec les modèles de deep learning.

## Données

Jeu de données Kaggle (vols aux États-Unis, 2015) : `flights.csv`,
`airlines.csv` et `airports.csv`. Ces fichiers ne sont pas inclus dans ce
dépôt (trop volumineux). Les modèles de deep learning entraînés (`dl_models/`)
ne sont pas inclus non plus.

Après nettoyage : 262 793 vols et 32 variables, séparés en 210 234 vols
d'entraînement et 52 559 vols de test (80 % / 20 %).

Répartition des classes : 196 459 vols à l'heure, 24 005 vols en retard de
15 à 30 minutes (environ 9 %), 42 329 vols en retard de plus de 30 minutes.

## Résultats du machine learning (52 559 vols de test)

| Modèle | Accuracy | ROC-AUC |
|---|---|---|
| Régression logistique | 0,8154 | 0,8950 |
| Random Forest | 0,8867 | 0,9220 |
| SVM (RBF, 15 000 lignes) | 0,8298 | 0,9078 |
| XGBoost | 0,8930 | 0,9307 |
| XGBoost (GridSearchCV) | 0,8903 | 0,9256 |

Meilleur modèle : XGBoost (ROC-AUC 0,9307). Le GridSearchCV n'a pas amélioré
XGBoost.

## Résultats du deep learning (jeu de test de 5 000 vols)

| Modèle | Accuracy | ROC-AUC |
|---|---|---|
| MLP (Optuna) | 0,5916 | 0,7213 |
| LSTM | 0,5772 | 0,7074 |
| GRU | 0,5832 | 0,7113 |
| Transformer | 0,5758 | 0,7154 |

Les jeux de test du machine learning et du deep learning n'ont pas la même
taille : leurs scores ne sont pas strictement comparables.

## Limites

- `DEPARTURE_DELAY` est la variable la plus importante (44 %) : le modèle
  prédit le retard à l'arrivée une fois le retard au départ connu.
- La classe « retard entre 15 et 30 minutes » est la plus difficile à
  prédire (environ 9 % des vols).
- Le SVM n'est entraîné que sur 15 000 lignes (contrainte de temps de calcul).

## Lancer le projet

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib torch
python pipeline_automatique_final1.py --flights flights.csv --airlines airlines.csv --airports airports.csv
```

Les notebooks ont été exécutés sur Google Colab.
