"""Télécharge les fichiers de la compétition Kaggle Home Credit Default Risk.

Source : https://www.kaggle.com/competitions/home-credit-default-risk
Les fichiers sont enregistrés dans data/raw/. Ceux déjà présents ne sont pas retéléchargés.

Prérequis (une seule fois) :
1. accepter les règles de la compétition sur
   https://www.kaggle.com/competitions/home-credit-default-risk/rules
2. enregistrer un jeton d'API (https://www.kaggle.com/settings/api) dans ~/.kaggle/access_token

Utilisation : python src/download_data.py
"""

import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "home-credit-default-risk"
FICHIERS = {
    "application_train.csv": "une ligne par demande de prêt, avec la cible TARGET",
    "bureau.csv": "crédits des clients chez d'autres organismes",
    "HomeCredit_columns_description.csv": "description de chaque colonne",
}

# data/raw/ à la racine du projet, quel que soit le dossier d'où l'on lance le script
DOSSIER_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def telecharger(api: KaggleApi, nom: str) -> bool:
    """Télécharge un fichier de la compétition.

    Renvoie True si le fichier vient d'être téléchargé, False s'il était déjà là.
    """
    if (DOSSIER_RAW / nom).exists():
        return False
    # Kaggle envoie les gros fichiers compressés (nom.csv.zip) : unzip=True extrait
    # le CSV puis supprime l'archive.
    api.competition_download_file(COMPETITION, nom, path=str(DOSSIER_RAW), quiet=True, unzip=True)
    return True


def main() -> int:
    DOSSIER_RAW.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()  # s'arrête avec un message d'aide si aucun jeton n'est trouvé
    echecs = []

    for i, (nom, description) in enumerate(FICHIERS.items(), start=1):
        print(f"[{i}/{len(FICHIERS)}] {nom:<36}", end="", flush=True)
        try:
            nouveau = telecharger(api, nom)
        except Exception as erreur:  # jeton invalide, règles non acceptées, coupure réseau…
            print(f"ÉCHEC ({erreur})")
            echecs.append(nom)
            continue
        taille = (DOSSIER_RAW / nom).stat().st_size
        taille_lisible = f"{taille / 1e6:.0f} Mo" if taille >= 1e6 else f"{taille / 1e3:.0f} Ko"
        etat = "téléchargé" if nouveau else "déjà présent"
        print(f"{etat} ({taille_lisible}) : {description}")

    if echecs:
        print(
            f"\n{len(echecs)} échec(s) : {', '.join(echecs)}.\n"
            "Vérifiez que les règles de la compétition sont acceptées et que le jeton est valide, "
            "puis relancez le script."
        )
        return 1
    print(f"\nTerminé : {len(FICHIERS)} fichiers dans {DOSSIER_RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
