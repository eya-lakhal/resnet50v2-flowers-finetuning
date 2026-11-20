# Interface Streamlit : Fine-tuning ResNet50V2 (tf_flowers)

## Lancer
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Contenu
- `app.py` : application (12 pages = les étapes du notebook + inférence).
- `data/results.json` et `assets/` : courbes, résultats d'optimisation, métriques et figures **extraits du notebook Colab**
  (régénérables : `python extract_data.py Projet_FineTuning.ipynb`).
- `models/` : y déposer `flowers_resnet50v2_finetuned.keras` (ou le charger depuis la barre latérale).

## Modèle pour l'inférence
Dans Colab, panneau Fichiers → télécharger `flowers_resnet50v2_finetuned.keras`, puis le placer dans `models/`.
Sans lui, tout fonctionne sauf la prédiction en direct et l'exploration en direct des filtres (les figures du notebook restent affichées).
Le modèle a été sauvegardé avec Keras 3 : utilisez TensorFlow ≥ 2.16.

## Ce qui est interactif
Data augmentation en direct,
heatmap Grid Search, Random Search, validation croisée, explorateur d'essais filtrable, courbes zoomables, rapport par classe,
inférence (upload / caméra) et feature maps de n'importe quelle couche.
