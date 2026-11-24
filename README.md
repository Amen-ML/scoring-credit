# Scoring crédit : prédire le risque de défaut d'un emprunteur

Projet de classification sur les demandes de prêt réelles de Home Credit (compétition Kaggle *Home Credit Default Risk*) : de la régression logistique à XGBoost, avec un seuil de décision choisi selon le coût d'une erreur et des décisions expliquées avec SHAP.

## Question

À partir du dossier de demande de prêt d'un client (revenus, montant demandé, emploi, famille, historique de crédit), peut-on estimer sa probabilité d'avoir des difficultés de remboursement ? Et à partir de quelle probabilité faut-il refuser le prêt pour que la banque perde le moins d'argent ?

Pour qui : un organisme de crédit qui prête à des clients ayant peu ou pas d'historique bancaire, et qui veut une décision rapide, chiffrée et explicable au client.

## Données

- **Source** : compétition Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (2018), données fournies par Home Credit, un organisme de crédit à la consommation.
- **Fichier principal** : `application_train.csv`, 307 511 demandes de prêt et 122 colonnes (166 Mo). Une ligne correspond à un prêt et à un client.
- **Variable à prédire** : `TARGET` = 1 si le client a eu un retard de paiement sur au moins une des premières échéances du prêt (24 825 prêts, soit 8,07 %), 0 sinon.
- **Fichier complémentaire** : `bureau.csv`, 1 716 428 lignes décrivant les crédits des clients chez d'autres organismes. Un client peut y avoir plusieurs lignes, et 86 % des demandes y ont au moins un crédit. Le fichier sera agrégé en une ligne par client.
- **Dictionnaire des variables** : `HomeCredit_columns_description.csv` décrit les 219 colonnes de la compétition (encodé en latin-1).
- **Accès** : les données sont soumises aux règles de la compétition Kaggle et ne sont pas redistribuées dans ce dépôt. `src/download_data.py` les télécharge dans `data/raw/` avec l'API Kaggle.

## Périmètre

- **Uniquement `application_train.csv` (+ `bureau.csv`)** : le fichier `application_test.csv` de Kaggle n'a pas d'étiquettes, il ne permet donc pas de mesurer l'erreur. Je découpe `application_train.csv` en un jeu d'apprentissage (80 %) et un jeu de test (20 %).
- **Découpage aléatoire stratifié** : contrairement au projet de prix immobiliers, les demandes ne sont pas datées, il n'y a donc pas de découpage temporel possible. La stratification garde la même proportion de défauts (8 %) dans les deux jeux. Le jeu de test n'est utilisé qu'une seule fois, à la fin.
- **Variables exclues pour des raisons éthiques** : le sexe (`CODE_GENDER`) n'entre pas dans le modèle. Refuser un crédit en fonction du sexe est discriminatoire, et l'AI Act européen classe le scoring crédit des particuliers parmi les systèmes d'IA « à haut risque », qui doivent être explicables et non discriminatoires. Retirer la colonne ne suffit pas, car d'autres variables reflètent le sexe (95 % des chauffeurs sont des hommes, 97 % des comptables sont des femmes) : un contrôle d'équité du modèle final est donc prévu.
- **Les autres fichiers de la compétition** (anciennes demandes chez Home Credit, historiques de paiements mensuels) sont laissés de côté : `bureau.csv` suffit à montrer l'agrégation de plusieurs lignes par client sans dépasser la mémoire d'un ordinateur portable.

## Démarche prévue

1. Télécharger les données et les explorer : déséquilibre des classes, valeurs manquantes, valeurs aberrantes, lien entre chaque variable et le défaut
2. Découper en apprentissage / test (80 / 20, stratifié), puis mettre le test de côté
3. Baselines : un modèle qui prédit « personne ne fait défaut » (pour montrer le piège de l'accuracy), puis la moyenne des scores externes (`EXT_SOURCE_1` à `3`) utilisée directement comme score
4. Régression logistique : imputation des valeurs manquantes, standardisation et one-hot encoding dans un `Pipeline` scikit-learn ; lecture des coefficients en odds ratios
5. Arbre de décision : critère de Gini, profondeur, sur-apprentissage, lecture des règles
6. Random Forest puis XGBoost : bagging contre boosting, réglage des hyperparamètres par validation croisée stratifiée, early stopping
7. Nouvelles variables : ratios (crédit / revenu, annuité / revenu, âge, ancienneté dans l'emploi) et agrégats de l'historique de crédit (`bureau.csv`)
8. Déséquilibre des classes : comparer l'absence de correction, la pondération des classes (`class_weight`, `scale_pos_weight`) et le sous-échantillonnage
9. Seuil de décision : choisir le seuil qui minimise le coût métier, vérifier la calibration des probabilités
10. Interprétabilité et équité : variables les plus importantes et explication d'une décision individuelle avec SHAP, puis comparaison des taux de refus et des erreurs entre hommes et femmes (le sexe sert à l'audit, jamais à la décision)
11. Application Streamlit : saisir le dossier d'un client et obtenir sa probabilité de défaut, la décision (accord ou refus) et les raisons principales de cette décision

## Mesure de l'erreur

- **Pourquoi pas l'accuracy** : avec 8 % de défauts, un modèle qui accepte tout le monde a raison 92 fois sur 100, sans détecter un seul défaut.
- **Métrique principale : ROC AUC.** C'est la probabilité qu'un client en défaut tiré au hasard reçoive un score de risque plus élevé qu'un bon client tiré au hasard (0,5 = hasard, 1 = classement parfait). Elle ne dépend pas du seuil choisi, et c'est la métrique officielle de la compétition.
- **Métrique complémentaire : PR AUC** (aire sous la courbe précision-rappel), plus sensible à la qualité du modèle sur la classe rare.
- **Coût métier au seuil choisi** : accorder un prêt à un client qui fera défaut (faux négatif) fait perdre une grande partie du capital prêté, alors que refuser un bon client (faux positif) ne fait perdre que la marge sur les intérêts. Hypothèse de départ : **un faux négatif coûte 10 fois plus qu'un faux positif**, avec une analyse de sensibilité pour des rapports de 5 et de 20.
- Au seuil retenu : matrice de confusion, rappel (part des défauts détectés) et précision (part des refus justifiés).

## Notions et outils

- **Notions** : classification binaire, régression logistique (sigmoïde, log-loss, odds ratios), arbre de décision (Gini, profondeur), méthodes d'ensemble (bagging avec Random Forest, boosting avec XGBoost), validation croisée stratifiée, réglage des hyperparamètres, déséquilibre des classes, courbes ROC et précision-rappel, matrice de confusion, seuil de décision et coût métier, calibration, valeurs SHAP, équité d'un modèle, création de variables, pipeline scikit-learn
- **Outils** : Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, SHAP, statsmodels, Jupyter, API Kaggle, Streamlit, Git

## Structure du projet

```
scoring-credit/
├── data/raw/          # fichiers Kaggle téléchargés (non versionnés)
├── data/processed/    # tables préparées (non versionnées)
├── notebooks/         # exploration, puis un notebook par famille de modèles
├── src/
│   └── download_data.py   # téléchargement des fichiers Kaggle
├── models/            # modèle entraîné utilisé par l'application
├── requirements-dev.txt   # environnement complet : notebooks, entraînement
└── README.md
```

## Installation

1. Créer un compte sur [Kaggle](https://www.kaggle.com) et accepter les règles de la compétition sur la page [Rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules).
2. Générer un jeton d'API dans [Settings → API](https://www.kaggle.com/settings/api) (« Generate New Token ») et l'enregistrer dans `~/.kaggle/access_token`.
3. Installer le projet et télécharger les données :

```bash
git clone https://github.com/Amen-ML/scoring-credit.git
cd scoring-credit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python src/download_data.py   # télécharge application_train.csv, bureau.csv et la description des colonnes
```

## Résultats

À venir.

## Inspirations

- La compétition Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk), pour les données et la métrique
- [levist7/Credit_Risk_Modelling](https://github.com/levist7/Credit_Risk_Modelling) et [shawn-y-sun/Credit_Risk_Model_LoanDefaults](https://github.com/shawn-y-sun/Credit_Risk_Model_LoanDefaults), pour la modélisation de la probabilité de défaut par régression logistique
