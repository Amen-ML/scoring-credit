"""Prépare les données Home Credit pour la modélisation.

Applique les règles décidées dans notebooks/01_exploration.ipynb (section 10) :
- découpage apprentissage / test 80 / 20, stratifié, identique à celui du notebook 01 ;
- retrait des colonnes redondantes (immeuble) et presque constantes ;
- dates en années, code DAYS_EMPLOYED = 365 243 → valeur manquante + indicateur SANS_EMPLOI ;
- nouvelles variables : 4 ratios sur les montants et des agrégats de bureau.csv par client.

Ce script n'applique que des règles fixées à l'avance, ligne par ligne ou client par client.
Tout ce qui s'apprend sur les données (médiane pour l'imputation, standardisation,
regroupement des modalités rares) se fait plus tard dans les Pipeline scikit-learn,
ajustés sur le seul jeu d'apprentissage : aucune information du test ne fuit.

Entrées : data/raw/application_train.csv et data/raw/bureau.csv
Sorties : data/processed/train.parquet (80 %) et data/processed/test.parquet (20 %)

Utilisation : python src/prepare_data.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_RAW = RACINE / "data" / "raw"
DOSSIER_PROCESSED = RACINE / "data" / "processed"

# Découpage : mêmes paramètres que dans le notebook 01
PART_TEST = 0.2
GRAINE = 42

# Colonnes à ne jamais donner au modèle. CODE_GENDER est conservé seulement pour
# le contrôle d'équité de la fin du projet (le sexe sert à l'audit, jamais à la décision).
COLONNES_HORS_MODELE = ["SK_ID_CURR", "TARGET", "CODE_GENDER"]

# Indicateurs dont la valeur minoritaire concerne moins de 0,1 % des prêts (notebook 01, section 4.4)
QUASI_CONSTANTES = [
    "FLAG_MOBIL", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
]

# Colonnes sur l'immeuble en _MODE qui n'ont pas d'équivalent _AVG : on les garde
MODE_A_GARDER = ["TOTALAREA_MODE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
                 "EMERGENCYSTATE_MODE"]

JOURS_EN_ANNEES = {
    "DAYS_BIRTH": "AGE",
    "DAYS_EMPLOYED": "ANCIENNETE_EMPLOI",
    "DAYS_REGISTRATION": "ANNEES_DEPUIS_CHGT_ENREGISTREMENT",
    "DAYS_ID_PUBLISH": "ANNEES_DEPUIS_CHGT_PIECE_IDENTITE",
    "DAYS_LAST_PHONE_CHANGE": "ANNEES_DEPUIS_CHGT_TELEPHONE",
}
CODE_SANS_EMPLOI = 365243

# Nouvelles variables (étape 7 du README) : on mesurera ce qu'elles apportent aux modèles
VARIABLES_RATIOS = [
    "RATIO_CREDIT_REVENU",    # poids du crédit par rapport au revenu annuel
    "RATIO_ANNUITE_REVENU",   # taux d'effort : part du revenu consacrée au remboursement
    "RATIO_CREDIT_ANNUITE",   # nombre approximatif d'échéances
    "RATIO_BIEN_CREDIT",      # part du crédit couverte par le prix du bien
]
VARIABLES_BUREAU = [
    "BUREAU_AUCUN_HISTORIQUE",              # 1 si aucun crédit déclaré ailleurs
    "BUREAU_NB_CREDITS",
    "BUREAU_NB_ACTIFS",
    "BUREAU_NB_RECENTS",                    # ouverts dans l'année précédant la demande
    "BUREAU_NB_VENDUS_OU_IRRECOUVRABLES",   # statut « Sold » ou « Bad debt »
    "BUREAU_NB_EN_RETARD",                  # crédits avec un retard en cours
    "BUREAU_RETARD_MAX_JOURS",
    "BUREAU_MONTANT_RETARD_MAX",
    "BUREAU_NB_PROLONGATIONS",
    "BUREAU_DETTE_EN_COURS",                # dette restante sur les crédits actifs
    "BUREAU_RATIO_DETTE_REVENU",
    "BUREAU_ANNEES_DEPUIS_DERNIER",         # ancienneté du dernier crédit ouvert
    "BUREAU_ANNEES_DEPUIS_PREMIER",         # ancienneté du premier crédit ouvert
]
VARIABLES_NOUVELLES = VARIABLES_RATIOS + VARIABLES_BUREAU


def decouper(demandes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage 80 / 20 stratifié sur TARGET, identique à celui du notebook 01."""
    return train_test_split(demandes, test_size=PART_TEST, stratify=demandes["TARGET"],
                            random_state=GRAINE)


def nettoyer(demandes: pd.DataFrame) -> pd.DataFrame:
    """Retire les colonnes inutiles et recode les dates et les valeurs aberrantes."""
    redondantes = [c for c in demandes.columns
                   if c.endswith("_MEDI") or (c.endswith("_MODE") and c not in MODE_A_GARDER)]
    demandes = demandes.drop(columns=redondantes + QUASI_CONSTANTES)

    # 365 243 jours (1 000 ans) est le code des clients sans emploi (retraités, chômeurs)
    demandes["SANS_EMPLOI"] = (demandes["DAYS_EMPLOYED"] == CODE_SANS_EMPLOI).astype(int)
    demandes["DAYS_EMPLOYED"] = demandes["DAYS_EMPLOYED"].replace(CODE_SANS_EMPLOI, np.nan)

    # Jours avant la demande (négatifs) → années (positives)
    for jours, annees in JOURS_EN_ANNEES.items():
        demandes[annees] = demandes[jours].abs() / 365.25
    demandes = demandes.drop(columns=list(JOURS_EN_ANNEES))

    # Oui / non → 1 / 0
    for colonne in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        demandes[colonne] = demandes[colonne].map({"Y": 1, "N": 0})
    return demandes


def ajouter_ratios(demandes: pd.DataFrame) -> pd.DataFrame:
    """Ratios entre les montants (le revenu et le crédit ne sont jamais nuls)."""
    demandes["RATIO_CREDIT_REVENU"] = demandes["AMT_CREDIT"] / demandes["AMT_INCOME_TOTAL"]
    demandes["RATIO_ANNUITE_REVENU"] = demandes["AMT_ANNUITY"] / demandes["AMT_INCOME_TOTAL"]
    demandes["RATIO_CREDIT_ANNUITE"] = demandes["AMT_CREDIT"] / demandes["AMT_ANNUITY"]
    demandes["RATIO_BIEN_CREDIT"] = demandes["AMT_GOODS_PRICE"] / demandes["AMT_CREDIT"]
    return demandes


def agreger_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Résume les crédits d'un client chez d'autres organismes en une ligne par client.

    Chaque client n'est résumé qu'à partir de ses propres crédits, sans utiliser TARGET :
    le calcul est le même pour un client de l'apprentissage, du test ou un nouveau client.
    """
    actif = bureau["CREDIT_ACTIVE"] == "Active"
    credits = bureau.assign(
        actif=actif,
        recent=bureau["DAYS_CREDIT"] > -365,
        vendu_ou_irrecouvrable=bureau["CREDIT_ACTIVE"].isin(["Sold", "Bad debt"]),
        en_retard=bureau["CREDIT_DAY_OVERDUE"] > 0,
        annees_depuis_ouverture=bureau["DAYS_CREDIT"].abs() / 365.25,
        # Dette des seuls crédits actifs ; quelques dettes négatives (anomalies) ramenées à 0
        dette_active=bureau["AMT_CREDIT_SUM_DEBT"].clip(lower=0).where(actif, 0),
    )
    par_client = credits.groupby("SK_ID_CURR")
    return pd.DataFrame({
        "BUREAU_NB_CREDITS": par_client.size(),
        "BUREAU_NB_ACTIFS": par_client["actif"].sum(),
        "BUREAU_NB_RECENTS": par_client["recent"].sum(),
        "BUREAU_NB_VENDUS_OU_IRRECOUVRABLES": par_client["vendu_ou_irrecouvrable"].sum(),
        "BUREAU_NB_EN_RETARD": par_client["en_retard"].sum(),
        "BUREAU_RETARD_MAX_JOURS": par_client["CREDIT_DAY_OVERDUE"].max(),
        "BUREAU_MONTANT_RETARD_MAX": par_client["AMT_CREDIT_MAX_OVERDUE"].max(),
        "BUREAU_NB_PROLONGATIONS": par_client["CNT_CREDIT_PROLONG"].sum(),
        "BUREAU_DETTE_EN_COURS": par_client["dette_active"].sum(),
        "BUREAU_ANNEES_DEPUIS_DERNIER": par_client["annees_depuis_ouverture"].min(),
        "BUREAU_ANNEES_DEPUIS_PREMIER": par_client["annees_depuis_ouverture"].max(),
    })


def ajouter_bureau(demandes: pd.DataFrame, agregats: pd.DataFrame) -> pd.DataFrame:
    """Joint les agrégats de bureau.csv ; un client sans historique a 0 crédit, 0 dette, 0 retard."""
    demandes = demandes.join(agregats, on="SK_ID_CURR")
    sans_historique = demandes["BUREAU_NB_CREDITS"].isna()
    demandes["BUREAU_AUCUN_HISTORIQUE"] = sans_historique.astype(int)

    # Comptes, dettes et retards manquants → 0 (pas de crédit, ou aucun retard enregistré).
    # Les anciennetés restent manquantes : il n'y a pas de crédit dont mesurer l'âge.
    a_zero = [c for c in agregats.columns if not c.startswith("BUREAU_ANNEES")]
    demandes[a_zero] = demandes[a_zero].fillna(0)
    demandes["BUREAU_RATIO_DETTE_REVENU"] = demandes["BUREAU_DETTE_EN_COURS"] / demandes["AMT_INCOME_TOTAL"]
    return demandes


def preparer(demandes: pd.DataFrame, agregats: pd.DataFrame) -> pd.DataFrame:
    """Applique toutes les règles à un jeu de demandes de prêt."""
    demandes = nettoyer(demandes.copy())
    demandes = ajouter_ratios(demandes)
    return ajouter_bureau(demandes, agregats)


def main() -> int:
    debut = time.perf_counter()
    DOSSIER_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Lecture des fichiers bruts…")
    demandes = pd.read_csv(DOSSIER_RAW / "application_train.csv")
    bureau = pd.read_csv(DOSSIER_RAW / "bureau.csv")
    print(f"  application_train.csv : {len(demandes):,} demandes, {demandes.shape[1]} colonnes")
    print(f"  bureau.csv            : {len(bureau):,} crédits")

    train, test = decouper(demandes)
    agregats = agreger_bureau(bureau)

    for nom, jeu in [("train", train), ("test", test)]:
        jeu = preparer(jeu, agregats)
        chemin = DOSSIER_PROCESSED / f"{nom}.parquet"
        jeu.to_parquet(chemin, index=False)
        print(f"\n{chemin.relative_to(RACINE)} : {len(jeu):,} prêts, {jeu.shape[1]} colonnes, "
              f"taux de défaut {jeu['TARGET'].mean():.2%}")

    nb_explicatives = jeu.shape[1] - len(COLONNES_HORS_MODELE)
    print(f"\nVariables explicatives : {nb_explicatives} "
          f"(dont {len(VARIABLES_RATIOS)} ratios et {len(VARIABLES_BUREAU)} agrégats de bureau.csv) ; "
          f"hors modèle : {', '.join(COLONNES_HORS_MODELE)}")
    print(f"Terminé en {time.perf_counter() - debut:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
