"""
Interface Streamlit du projet « Fine-tuning de ResNet50V2 – classification de fleurs (tf_flowers) ».

Les courbes, tableaux et métriques proviennent du notebook exécuté sur Google Colab
(extraits dans data/results.json par extract_data.py). La data augmentation est calculée en direct. L'inférence, les filtres et les cartes de
caractéristiques en direct nécessitent le modèle sauvegardé (.keras) ; sinon l'application
affiche les figures du notebook.

Lancement :  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
MODELS_DIR = ROOT / "models"
MODEL_NAME = "flowers_resnet50v2_finetuned.keras"
IMG_SIZE = (224, 224)

CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CLASSES_FR = {"daisy": "Marguerite", "dandelion": "Pissenlit", "roses": "Rose",
              "sunflowers": "Tournesol", "tulips": "Tulipe"}
CLASS_COLORS = {"daisy": "#8d99ae", "dandelion": "#e9c46a", "roses": "#d6336c",
                "sunflowers": "#f4a261", "tulips": "#7b2cbf"}
GREEN, TEAL, GOLD, CORAL, GREY = "#2a7f62", "#2a9d8f", "#e9c46a", "#e76f51", "#8d99ae"
MODEL_COLORS = [GREY, TEAL, GOLD, CORAL]

st.set_page_config(page_title="Fine-tuning ResNet50V2 · Fleurs", page_icon="🌼",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- style
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1250px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
  background: linear-gradient(120deg, rgba(42,127,98,.18), rgba(233,196,106,.20) 55%, rgba(214,51,108,.14));
  border: 1px solid rgba(42,127,98,.25);}
.hero h1 {margin: 0 0 .3rem 0; font-size: 2.1rem;}
.hero p {margin: 0; opacity: .85; font-size: 1.02rem;}
.card {padding: .9rem 1.1rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);
  background: rgba(128,128,128,.06); height: 100%;}
.card h4 {margin: 0 0 .25rem 0; font-size: .98rem;}
.card p {margin: 0; font-size: .88rem; opacity: .85;}
.step-chip {display:inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .78rem;
  font-weight: 600; background: rgba(42,127,98,.16); color: #2a7f62; margin-bottom: .4rem;}
.note {padding: .8rem 1rem; border-left: 4px solid #2a7f62; border-radius: 6px;
  background: rgba(42,127,98,.09); margin: .6rem 0 1rem 0;}
.warn {padding: .8rem 1rem; border-left: 4px solid #e76f51; border-radius: 6px;
  background: rgba(231,111,81,.10); margin: .6rem 0 1rem 0;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.07); padding: .7rem .9rem;
  border-radius: 12px; border: 1px solid rgba(128,128,128,.2);}
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0; padding: .4rem 1rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def fragment(fn):
    """Fragment Streamlit : les widgets de la démo ne relancent que la démo (interface plus fluide)."""
    deco = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    return deco(fn) if deco else fn


def plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:                                   # anciennes versions de Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_image(img, caption=None, **kw):
    try:
        st.image(img, caption=caption, width="stretch", **kw)
    except TypeError:
        st.image(img, caption=caption, use_container_width=True, **kw)


def asset(name, caption=None):
    p = ASSETS / f"{name}.png"
    if p.exists():
        show_image(str(p), caption)
    else:
        st.info(f"Figure « {name} » introuvable (relancez extract_data.py).")


def header(step: str, title: str, subtitle: str = ""):
    st.markdown(f'<span class="step-chip">{step}</span>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def note(text, kind="note"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def cards(items):
    cols = st.columns(len(items))
    for col, (t, d) in zip(cols, items):
        col.markdown(f'<div class="card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)


@st.cache_data
def load_results():
    with open(ROOT / "data" / "results.json", encoding="utf-8") as f:
        return json.load(f)


R = load_results()
HIST = {k: pd.DataFrame(v) for k, v in R["histories"].items()}


def curves(df: pd.DataFrame, title: str, offset: int = 0, boundary: int | None = None):
    """Perte, précision et learning rate par époque (interactif)."""
    ep = np.arange(1, len(df) + 1) + offset
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Perte (cross-entropy)", "Précision", "Learning rate"),
                        horizontal_spacing=0.07)
    for col, (a, b) in enumerate([("loss", "val_loss"), ("acc", "val_acc")], start=1):
        fig.add_trace(go.Scatter(x=ep, y=df[a], name="Entraînement", mode="lines+markers",
                                 line=dict(color=TEAL, width=2.5), legendgroup="tr", showlegend=col == 1), 1, col)
        fig.add_trace(go.Scatter(x=ep, y=df[b], name="Validation", mode="lines+markers",
                                 line=dict(color=CORAL, width=2.5), legendgroup="va", showlegend=col == 1), 1, col)
    fig.add_trace(go.Scatter(x=ep, y=df["lr"], name="Learning rate", mode="lines+markers",
                             line=dict(color=GOLD, width=2.5, shape="hv"), showlegend=False), 1, 3)
    fig.update_yaxes(type="log", row=1, col=3)
    if boundary is not None:
        for c in (1, 2, 3):
            fig.add_vline(x=boundary + 0.5, line_dash="dash", line_color=GREY, row=1, col=c)
    fig.update_xaxes(title_text="Époque")
    fig.update_layout(title=title, height=380, margin=dict(t=70, b=40, l=10, r=10),
                      legend=dict(orientation="h", y=-0.22))
    plot(fig)


def best_epoch(df):
    i = int(df["val_loss"].idxmin())
    return i + 1, df.loc[i]


# --------------------------------------------------------------------------- modèle (optionnel)
@st.cache_resource(show_spinner="Chargement du modèle (TensorFlow)…")
def load_keras_model(path: str):
    import keras  # noqa: WPS433  (import tardif : l'app fonctionne sans TensorFlow)
    return keras.saving.load_model(path, compile=False)


def get_model():
    """Retourne (modèle, message). Le modèle est cherché dans models/ ou fourni via la barre latérale."""
    p = MODELS_DIR / MODEL_NAME
    if not p.exists():
        cands = sorted(MODELS_DIR.glob("*.keras"))
        p = cands[0] if cands else None
    if p is None:
        return None, "Aucun modèle .keras trouvé"
    try:
        return load_keras_model(str(p)), f"Modèle chargé : {p.name}"
    except ModuleNotFoundError:
        return None, "TensorFlow/Keras non installé"
    except Exception as e:  # noqa: BLE001
        return None, f"Échec du chargement : {e}"


def prepare_image(pil: Image.Image) -> np.ndarray:
    return np.asarray(pil.convert("RGB").resize(IMG_SIZE), dtype="float32")


def predict(model, pil: Image.Image) -> np.ndarray:
    # Le modèle contient déjà Rescaling([-1,1]) : on lui donne des pixels bruts 0-255.
    return model.predict(prepare_image(pil)[None, ...], verbose=0)[0]


def base_of(model):
    import keras
    for l in model.layers:
        if isinstance(l, keras.Model) and "resnet" in l.name.lower():
            return l
    return None


def tile(arrs, ncols, pad=2):
    """Assemble des tableaux 2D/3D (déjà normalisés 0-1) en une mosaïque."""
    h, w = arrs[0].shape[:2]
    nrows = math.ceil(len(arrs) / ncols)
    ch = arrs[0].shape[2] if arrs[0].ndim == 3 else 1
    canvas = np.ones((nrows * (h + pad), ncols * (w + pad), ch), dtype="float32")
    for i, a in enumerate(arrs):
        r, c = divmod(i, ncols)
        a = a if a.ndim == 3 else a[..., None]
        canvas[r * (h + pad): r * (h + pad) + h, c * (w + pad): c * (w + pad) + w] = a
    return canvas if ch == 3 else canvas[..., 0]


def viridis(a):
    from matplotlib import cm
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return cm.viridis(a)[..., :3].astype("float32")


# --------------------------------------------------------------------------- image d'exemple (fleur dessinée)
@st.cache_data
def sample_flower(kind: str = "daisy", size: int = 320) -> Image.Image:
    img = Image.new("RGB", (size, size), (110, 160, 90))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(3)
    for _ in range(260):                                  # fond « herbe »
        x, y = rng.integers(0, size, 2)
        g = int(rng.integers(90, 190))
        d.line([x, y, x + int(rng.integers(-6, 6)), y - int(rng.integers(8, 26))], fill=(40, g, 50), width=2)
    cx = cy = size // 2
    petal, centre = {"daisy": ((250, 250, 245), (240, 190, 40)),
                     "sunflower": ((245, 190, 30), (90, 55, 25)),
                     "rose": ((205, 40, 90), (150, 20, 60))}[kind]
    n = 14
    for k in range(n):
        a = 2 * math.pi * k / n
        px, py = cx + 70 * math.cos(a), cy + 70 * math.sin(a)
        d.ellipse([px - 28, py - 28, px + 28, py + 28], fill=petal, outline=(200, 200, 190))
    d.ellipse([cx - 38, cy - 38, cx + 38, cy + 38], fill=centre)
    return img


# --------------------------------------------------------------------------- pages
def page_home():
    st.markdown("""
    <div class="hero"><h1>🌼 Fine-tuning d'un CNN pré-entraîné</h1>
    <p>ResNet50V2 (ImageNet) adapté à la classification de 5 espèces de fleurs — jeu de données <b>tf_flowers</b>
    (3 670 images). Parcourez chaque étape du projet : données, modèles, optimisation, évaluation et inférence.</p></div>
    """, unsafe_allow_html=True)

    m = {x["name"]: x for x in R["models"]}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meilleure accuracy (test)", f"{m['ResNet50V2 : fine-tuning']['test_acc']:.2f} %")
    c2.metric("Gain vs CNN from scratch",
              f"+{m['ResNet50V2 : fine-tuning']['test_acc'] - m['CNN from scratch']['test_acc']:.1f} pts")
    c3.metric("Images de test", f"{R['report']['n_test']}", help="Jamais utilisées pour choisir un hyperparamètre")
    c4.metric("Erreurs de test", f"{R['report']['n_errors']}")

    st.subheader("Démarche du projet")
    steps = [("1 · Configuration", "Hyperparamètres et justifications"),
             ("2 · Données", "70 / 15 / 15 %, 5 classes équilibrées"),
             ("3 · Augmentation", "Flip, rotation, zoom, contraste"),
             ("4 · CNN from scratch", "Modèle de référence"),
             ("5 · Transfert", "Phase 1 : base gelée"),
             ("6 · Fine-tuning", "Phase 2 : dégel partiel"),
             ("7 · Optimisation", "Grid, Random, validation croisée"),
             ("8 · Filtres", "Filtres et feature maps"),
             ("9 · Évaluation", "Métriques, confusion, erreurs"),
             ("10 · Conclusion", "Comparaison et limites"),
             ("11 · Inférence", "Testez le modèle")]
    for row in range(0, len(steps), 4):
        cards(steps[row:row + 4])
        st.write("")

    note("<b>Résultat clé.</b> Le transfert d'apprentissage fait passer l'accuracy de test de 69,9 % (CNN from scratch) à 88,0 % "
         "avec la seule tête entraînée (10 245 paramètres), puis à <b>92,9 %</b> avec le fine-tuning des 30 dernières couches. "
         "La configuration « optimisée » par recherche (88,9 %) <i>ne fait pas mieux</i> que la configuration raisonnée : "
         "voir l'étape 7 pour l'explication.")

    st.subheader("Comparaison des 4 approches (jeu de test)")
    df = pd.DataFrame(R["models"])
    fig = go.Figure(go.Bar(x=df["name"], y=df["test_acc"], marker_color=MODEL_COLORS,
                           text=[f"{v:.2f} %" for v in df["test_acc"]], textposition="outside"))
    fig.update_layout(yaxis=dict(range=[0, 100], title="Accuracy test (%)"), height=360,
                      margin=dict(t=20, b=10, l=10, r=10))
    plot(fig)


def page_config():
    header("Étape 1", "Configuration et hyperparamètres",
           "Tous les réglages sont regroupés au même endroit dans le notebook (reproductibilité : SEED = 42).")
    hp = pd.DataFrame([
        ("IMG_SIZE", "224 × 224", "Taille d'entrée attendue par ResNet50V2"),
        ("Optimiseur", "Adam", "Taux d'apprentissage adaptatif par paramètre + momentum"),
        ("BATCH_SIZE", 32, "Mini-batch : compromis entre le bruit du SGD et le coût du batch complet"),
        ("EPOCHS_HEAD", 10, "Époques max. phase 1 (base gelée)"),
        ("EPOCHS_FT", 15, "Époques max. phase 2 (fine-tuning)"),
        ("LR_HEAD", "1e-3", "La tête est initialisée aléatoirement : elle peut apprendre vite"),
        ("LR_FT", "1e-5", "Petits pas pour éviter l'oubli catastrophique des poids ImageNet"),
        ("L2_LAMBDA (λ)", "1e-4", "Pénalité légère sur la couche de sortie"),
        ("DROPOUT_RATE", 0.3, "Régularisation modérée (la base pré-entraînée est déjà robuste)"),
        ("FINE_TUNE_LAYERS", 30, "Nombre de couches de la fin de la base que l'on dégèle"),
        ("PATIENCE", 3, "Early stopping : tolère de petites fluctuations de la val_loss"),
    ], columns=["Hyperparamètre", "Valeur", "Justification"]).astype({"Valeur": str})
    table(hp)

    st.subheader("Callbacks communs à tous les modèles")
    cards([("EarlyStopping", "Surveille <code>val_loss</code>, patience 3, restaure les meilleurs poids."),
           ("ReduceLROnPlateau", "Divise le learning rate par 5 si la <code>val_loss</code> stagne (min 1e-7)."),
           ("Comparaison équitable", "Mêmes fonctions de compilation, callbacks et métriques pour chaque modèle.")])


def page_data():
    header("Étape 2", "Chargement et préparation des données",
           "tf_flowers : 3 670 photos JPEG de taille variable, 5 classes.")
    d = R["data"]
    df = pd.DataFrame({"Classe": d["classes"], "Train": d["train"], "Validation": d["val"], "Test": d["test"]})
    tot = {k: sum(d[k]) for k in ("train", "val", "test")}
    c = st.columns(4)
    c[0].metric("Total", sum(tot.values()))
    c[1].metric("Entraînement", tot["train"], f"{tot['train'] / sum(tot.values()):.0%}")
    c[2].metric("Validation", tot["val"], f"{tot['val'] / sum(tot.values()):.0%}")
    c[3].metric("Test", tot["test"], f"{tot['test'] / sum(tot.values()):.0%}")

    t1, t2, t3 = st.tabs(["Répartition", "Aperçu des images", "Pipeline"])
    with t1:
        mode = st.radio("Affichage", ["Effectifs", "Pourcentage par classe"], horizontal=True)
        fig = go.Figure()
        for part, col in zip(["Train", "Validation", "Test"], [TEAL, GOLD, CORAL]):
            y = df[part] if mode == "Effectifs" else df[part] / df[["Train", "Validation", "Test"]].sum(axis=1) * 100
            fig.add_trace(go.Bar(name=part, x=df["Classe"], y=y, marker_color=col))
        fig.update_layout(barmode="group", height=360, margin=dict(t=10, b=10),
                          yaxis_title="Nombre d'images" if mode == "Effectifs" else "% de la classe")
        plot(fig)
        table(df)
        st.caption("Découpage aléatoire reproductible : la répartition des classes reste équilibrée dans les trois parties.")
    with t2:
        asset("samples", "Neuf images du jeu d'entraînement (tailles variables avant redimensionnement)")
    with t3:
        st.markdown("""
        1. **Redimensionnement** en 224 × 224 (taille attendue par ResNet50V2) ;
        2. **One-hot encoding** de l'étiquette (ex. `[1, 0, 0, 0, 0]` pour *daisy*) ;
        3. **Cache** en mémoire (les images ne sont décodées qu'une fois) ;
        4. **Mélange** (shuffle 1000) uniquement sur l'entraînement ;
        5. **Batch** de 32 + `prefetch` pour recouvrir calcul et chargement.
        """)
        st.code("Forme d'un batch d'images : (32, 224, 224, 3)\nForme d'un batch d'étiquettes : (32, 5)  (one-hot)", language="text")


def demo_augmentation(img: Image.Image, flip_on, rot, zoom, contrast, seed):
    rng = np.random.default_rng(seed)
    base = img.convert("RGB").resize((224, 224))
    outs = []
    for _ in range(9):
        im = base
        if flip_on and rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if rot > 0:
            im = im.rotate(float(rng.uniform(-rot, rot)) * 360, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        if zoom > 0:
            z = 1 + float(rng.uniform(-zoom, zoom))
            w = int(224 / z)
            left = (224 - w) // 2
            im = im.crop((left, left, left + w, left + w)).resize((224, 224), Image.BILINEAR) if z >= 1 else \
                _zoom_out(im, z)
        if contrast > 0:
            im = ImageEnhance.Contrast(im).enhance(1 + float(rng.uniform(-contrast, contrast)))
        outs.append(np.asarray(im))
    return outs


def _zoom_out(im, z):
    w = int(224 * z)
    small = im.resize((w, w), Image.BILINEAR)
    canvas = Image.new("RGB", (224, 224), (0, 0, 0))
    canvas.paste(small, ((224 - w) // 2, (224 - w) // 2))
    return canvas


def page_augmentation():
    header("Étape 3", "Data augmentation",
           "Transformations aléatoires appliquées uniquement pendant l'entraînement : le modèle ne voit jamais deux fois exactement la même image.")
    cards([("RandomFlip", "Retournement horizontal"), ("RandomRotation(0.1)", "± 10 % d'un tour (± 36°)"),
           ("RandomZoom(0.1)", "Zoom de ± 10 %"), ("RandomContrast(0.1)", "Contraste ± 10 %")])
    st.write("")
    t1, t2 = st.tabs(["Essayer en direct", "Résultat du notebook"])
    with t1:
        c1, c2 = st.columns([1, 2.2])
        with c1:
            src = st.radio("Image source", ["Marguerite (exemple)", "Tournesol (exemple)", "Rose (exemple)", "Mon image"])
            up = st.file_uploader("Votre image", type=["jpg", "jpeg", "png"]) if src == "Mon image" else None
            flip_on = st.checkbox("Flip horizontal", True)
            rot = st.slider("Rotation (fraction de tour)", 0.0, 0.5, 0.1, 0.01)
            zoom = st.slider("Zoom (±)", 0.0, 0.5, 0.1, 0.01)
            contrast = st.slider("Contraste (±)", 0.0, 0.6, 0.1, 0.01)
            seed = st.number_input("Graine aléatoire", 0, 9999, int(st.session_state.get("aug_seed", 1)))
            if st.button("🎲 Nouveau tirage"):
                st.session_state["aug_seed"] = int(seed) + 1
                st.rerun()
        if src == "Mon image" and up is not None:
            img = Image.open(up)
        else:
            img = sample_flower({"Marguerite": "daisy", "Tournesol": "sunflower", "Rose": "rose"}.get(src.split(" ")[0], "daisy"))
        with c2:
            outs = demo_augmentation(img, flip_on, rot, zoom, contrast, int(seed))
            grid = np.concatenate([np.concatenate(outs[r * 3:(r + 1) * 3], axis=1) for r in range(3)], axis=0)
            show_image(grid, "Neuf versions augmentées de la même image")
    with t2:
        asset("augmentation", "Une même image après différentes augmentations (notebook)")
    note("La data augmentation est une <b>régularisation</b> : elle fait varier les exemples sans en collecter de nouveaux et réduit le sur-apprentissage.")


def page_baseline():
    header("Étape 4", "Modèle de référence : CNN from scratch",
           "Un petit CNN entraîné depuis zéro sert de point de comparaison pour mesurer l'apport du transfert d'apprentissage.")
    h = HIST["baseline"]
    be, row = best_epoch(h)
    c = st.columns(4)
    c[0].metric("Paramètres", "12 939 077")
    c[1].metric("Meilleure époque", be, f"arrêt à {len(h)}/15 (early stopping)")
    c[2].metric("Val. accuracy", f"{row['val_acc']:.2%}")
    c[3].metric("Test accuracy", "69.87 %", help="Test loss = 0.8242")
    t1, t2, t3 = st.tabs(["Architecture", "Courbes d'entraînement", "Analyse"])
    with t1:
        arch = pd.DataFrame([
            ("Input + Data augmentation + Rescaling(1/255)", "224×224×3", 0),
            ("Conv2D 32 (3×3, ReLU) + MaxPool 2×2", "112×112×32", 896),
            ("Conv2D 64 (3×3, ReLU) + MaxPool 2×2", "56×56×64", 18_496),
            ("Conv2D 128 (3×3, ReLU) + MaxPool 2×2", "28×28×128", 73_856),
            ("Flatten", "100 352", 0),
            ("Dense 128 (ReLU, L2 λ=1e-4)", "128", 12_845_184),
            ("Dropout 0.5", "128", 0),
            ("Dense 5 (softmax)", "5", 645)], columns=["Couche", "Sortie", "Paramètres"])
        table(arch)
        st.caption(f"Total : {int(arch['Paramètres'].sum()):,} paramètres, dont 99 % dans la couche Dense (Flatten → 128).".replace(",", " "))
    with t2:
        curves(h, "CNN from scratch")
    with t3:
        note("Le CNN apprend ses caractéristiques <b>à partir de zéro</b> sur ~2 500 images : la précision de validation plafonne "
             "autour de 65 % (74 % en entraînement) et les courbes stagnent à la fin (le learning rate est divisé par 5 à plusieurs reprises). "
             "C'est la base de comparaison de tout le projet.")


def page_transfer():
    header("Étape 5", "Transfert d'apprentissage – Phase 1 : base gelée",
           "ResNet50V2 pré-entraîné sur ImageNet sert d'extracteur de caractéristiques ; seule la nouvelle tête est entraînée.")
    t1, t2, t3 = st.tabs(["Pourquoi ResNet50V2 ?", "Architecture", "Résultats phase 1"])
    with t1:
        cards([("Connexions résiduelles", "y = F(x) + x : le gradient traverse les raccourcis, ce qui permet d'entraîner un réseau profond."),
               ("Pré-entraîné ImageNet", "190 couches, 23,5 M de paramètres : bords, textures, formes déjà appris."),
               ("Hors famille VGG", "Comme demandé dans le sujet du projet.")])
    with t2:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = go.Figure(go.Funnel(
                y=["Entrée 224×224×3", "Data augmentation (train seul)", "Rescaling [−1, 1]", "Base ResNet50V2 (gelée)",
                   "GlobalAveragePooling2D", "Dropout 0.3", "Dense 5 + softmax (L2)"],
                x=[100, 92, 84, 76, 50, 40, 30], textinfo="text", text=["", "", "", "7×7×2048", "2048", "2048", "5"],
                marker=dict(color=[GREY, GREY, GREY, TEAL, GOLD, GOLD, CORAL])))
            fig.update_layout(height=380, margin=dict(t=10, b=10))
            plot(fig)
        with c2:
            st.metric("Paramètres entraînables", "10 245", help="2048 × 5 + 5")
            st.metric("Paramètres figés", "23 564 800")
            st.markdown("**`GlobalAveragePooling2D`** remplace `Flatten` : 2 048 valeurs au lieu de 7×7×2048 = 100 352, "
                        "donc moins de paramètres et moins de sur-apprentissage.")
            st.markdown("**`base_model(x, training=False)`** : les couches BatchNorm restent en mode inférence.")
    with t3:
        h = HIST["phase1"]
        be, row = best_epoch(h)
        c = st.columns(3)
        c[0].metric("Meilleure époque", be)
        c[1].metric("Val. accuracy", f"{row['val_acc']:.2%}")
        c[2].metric("Test accuracy", "88.02 %", "+18.15 pts vs from scratch")
        curves(h, "Phase 1 : extraction de caractéristiques (base gelée)")
        note("Avec ~10 000 paramètres entraînés seulement, la précision de validation dépasse déjà largement celle du CNN from scratch : "
             "les caractéristiques d'ImageNet sont réutilisables.")


def page_finetune():
    header("Étape 6", "Phase 2 : Fine-tuning",
           "On dégèle les dernières couches de la base et on les adapte aux fleurs, avec un learning rate très faible.")
    t1, t2, t3 = st.tabs(["Quelles couches dégeler ?", "Courbes phases 1 + 2", "Analyse"])
    with t1:
        total = 190
        n = st.slider("Nombre de couches dégelées (sur 190)", 0, total, 30)
        frozen = total - n
        fig = go.Figure()
        fig.add_trace(go.Bar(y=["Base"], x=[frozen], orientation="h", name="Figées (génériques : bords, textures)",
                             marker_color=GREY))
        fig.add_trace(go.Bar(y=["Base"], x=[n], orientation="h", name="Entraînées (spécialisées)", marker_color=CORAL))
        fig.update_layout(barmode="stack", height=170, margin=dict(t=10, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=-0.5), xaxis=dict(range=[0, total]))
        plot(fig)
        st.caption(f"Les couches 0 à {frozen - 1} restent figées ; les couches {frozen} à {total - 1} sont dégelées "
                   "(sauf les BatchNorm, toujours figées).")
        st.markdown("**Valeurs connues dans le notebook**")
        table(pd.DataFrame([
            ("Phase 1 (0 couche)", 10_245, 23_564_800),
            ("10 couches dégelées", 3_420_165, 20_154_880),
            ("30 couches dégelées (configuration par défaut)", 14_436_357, 9_138_688)],
            columns=["Configuration", "Paramètres entraînables", "Paramètres figés"]).astype(
            {"Paramètres entraînables": str, "Paramètres figés": str}))
        cards([("Recompiler", "Obligatoire après tout changement de <code>trainable</code>."),
               ("LR très faible", "1e-5 au lieu de 1e-3 : évite l'oubli catastrophique."),
               ("BatchNorm figées", "Pratique recommandée pour la stabilité.")])
    with t2:
        both = pd.concat([HIST["phase1"], HIST["finetune"]], ignore_index=True)
        curves(both, "Phases 1 + 2 : extraction puis fine-tuning", boundary=len(HIST["phase1"]))
        st.caption("Ligne pointillée : début du fine-tuning (époque 11).")
    with t3:
        h = HIST["finetune"]
        be, row = best_epoch(h)
        c = st.columns(4)
        c[0].metric("Meilleure époque", be + len(HIST["phase1"]))
        c[1].metric("Val. accuracy", f"{row['val_acc']:.2%}")
        c[2].metric("Test accuracy", "92.92 %", "+4.90 pts vs base gelée")
        c[3].metric("Test loss", "0.2252", "-0.0829")
        note("Le fine-tuning apporte un gain supplémentaire plus modeste que le transfert. La précision d'entraînement monte "
             "jusqu'à ~97 % alors que celle de validation se stabilise vers 92,5 % : un léger sur-apprentissage, contenu par "
             "l'early stopping et la réduction du learning rate (jusqu'à 1e-7).")


def page_optimization():
    header("Étape 7", "Optimisation des hyperparamètres",
           "Valider de façon systématique les choix initiaux : Grid Search, Random Search, puis wrapper scikit-learn avec validation croisée.")
    with st.expander("Protocole utilisé", expanded=False):
        st.markdown("""
        1. Hyperparamètres explorés : **learning rate du fine-tuning**, **nombre de couches dégelées**, **dropout**.
        2. Chaque essai **repart des mêmes poids** (fin de la phase 1) : comparaison équitable.
        3. Essais **courts** (3 époques maximum) avec early stopping pour limiter le coût.
        4. Score = **précision de validation** ; le **test n'est jamais utilisé** pour choisir.
        5. La meilleure configuration est réentraînée complètement puis comparée à la configuration par défaut.
        """)
    grid = pd.DataFrame(R["grid"])
    rnd = pd.DataFrame(R["random"])
    t1, t2, t3, t4, t5 = st.tabs(["Grid Search", "Random Search", "Validation croisée (SciKeras)", "Fine-tuning final", "Tous les essais"])

    with t1:
        st.markdown("Grille : **3 learning rates × 3 nombres de couches = 9 essais** (dropout fixé à 0.3).")
        metric = st.radio("Métrique affichée", ["val_acc", "val_loss"], horizontal=True, key="grid_metric",
                          format_func=lambda m: "Précision de validation" if m == "val_acc" else "Perte de validation")
        pv = grid.pivot(index="n_layers", columns="lr", values=metric).sort_index().sort_index(axis=1, ascending=False)
        fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{c:.0e}" for c in pv.columns], y=[str(i) for i in pv.index],
                                   colorscale="Viridis" if metric == "val_acc" else "Viridis_r",
                                   text=[[f"{v:.3f}" for v in r] for r in pv.values], texttemplate="%{text}",
                                   hovertemplate="lr=%{x}<br>couches=%{y}<br>valeur=%{z:.4f}<extra></extra>"))
        fig.update_layout(height=360, xaxis_title="Learning rate du fine-tuning", yaxis_title="Couches dégelées",
                          margin=dict(t=10, b=10))
        plot(fig)
        best = grid.sort_values(["val_acc", "val_loss"], ascending=[False, True]).iloc[0]
        st.success(f"Meilleur essai : lr = {best['lr']:.0e}, {int(best['n_layers'])} couches → val_acc = {best['val_acc']:.4f}")
        note("Les 9 scores sont tous compris entre <b>87,5 % et 89,1 %</b> : l'écart (~1,6 pt) est de l'ordre de la variabilité entre "
             "exécutions sur 550 images de validation (1 image = 0,18 pt). Aucun hyperparamètre ne se détache nettement.")

    with t2:
        st.markdown("6 essais : lr log-uniforme dans [1e-6, 1e-3], couches ∈ {10, 20, 30, 50, 80}, dropout ∈ [0.2, 0.5].")
        fig = go.Figure(go.Scatter(
            x=rnd["lr"], y=rnd["val_acc"], mode="markers",
            marker=dict(size=rnd["dropout"] * 70, color=rnd["n_layers"], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Couches"), line=dict(width=1, color="white")),
            customdata=np.stack([rnd["n_layers"], rnd["dropout"], rnd["val_loss"]], axis=1),
            hovertemplate="lr=%{x:.2e}<br>val_acc=%{y:.4f}<br>couches=%{customdata[0]}<br>dropout=%{customdata[1]:.2f}"
                          "<br>val_loss=%{customdata[2]:.4f}<extra></extra>"))
        fig.update_layout(height=380, xaxis=dict(type="log", title="Learning rate (échelle log)"),
                          yaxis_title="Précision de validation", margin=dict(t=10, b=10))
        plot(fig)
        st.caption("Un point par essai ; taille = dropout, couleur = nombre de couches dégelées.")
        table(rnd.sort_values("val_acc", ascending=False).rename(columns={
            "lr": "Learning rate", "n_layers": "Couches", "dropout": "Dropout", "val_acc": "Val. acc.",
            "val_loss": "Val. loss", "seconds": "Durée (s)"}).round(5))

    with t3:
        cv = R["cv"]
        st.markdown("Le wrapper scikit-learn encapsule la **tête de classification** (Dropout + Dense softmax) sur les 2 048 "
                    "caractéristiques extraites par la base figée, avec validation croisée à 3 plis. "
                    "(`tensorflow.keras.wrappers.scikit_learn` a été supprimé depuis TensorFlow 2.12, d'où SciKeras / wrapper maison.)")
        c = st.columns(3)
        c[0].metric("Tête par défaut (val.)", f"{cv['default_head_val']:.2%}")
        c[1].metric("GridSearchCV (CV 3 plis)", f"{cv['grid']['cv_score']:.2%}", f"val. {cv['grid']['val_score']:.2%}", delta_color="off")
        c[2].metric("RandomizedSearchCV (CV 3 plis)", f"{cv['random']['cv_score']:.2%}", f"val. {cv['random']['val_score']:.2%}", delta_color="off")
        st.markdown(f"**GridSearchCV** ({cv['grid']['n_fits']} fits) → `{cv['grid']['params']}`  \n"
                    f"**RandomizedSearchCV** ({cv['random']['n_fits']} fits) → `{cv['random']['params']}`")
        top = pd.DataFrame(cv["random_top5"])
        fig = go.Figure(go.Bar(x=[f"lr={r.lr:g} · λ={r.l2:g}" for r in top.itertuples()], y=top["mean"],
                               error_y=dict(type="data", array=top["std"]), marker_color=TEAL,
                               text=[f"{v:.3f}" for v in top["mean"]], textposition="outside"))
        fig.update_layout(height=340, yaxis=dict(range=[0.8, 0.9], title="Précision CV (moyenne ± écart-type)"),
                          title="RandomizedSearchCV : 5 meilleures combinaisons", margin=dict(t=50, b=10))
        plot(fig)
        note("La validation croisée fournit une <b>moyenne et un écart-type</b> sur 3 plis : estimation plus fiable qu'un unique "
             "score de validation. Les trois meilleures combinaisons sont à l'intérieur d'un écart-type l'une de l'autre.")

    with t4:
        f = R["final"]
        cfg = f["best_cfg"]
        st.markdown(f"Meilleure configuration trouvée (Random Search) : **lr = {cfg['lr']:.2e}**, "
                    f"**{cfg['n_layers']} couches**, **dropout = {cfg['dropout']:.2f}**.")
        fig = go.Figure(go.Bar(x=["Configuration par défaut\n(lr 1e-5, 30 couches, 25 époques)", "Configuration optimisée\n(essais de 3 époques)"],
                               y=[f["val_default"] * 100, f["val_optimized"] * 100], marker_color=[GOLD, CORAL],
                               text=[f"{f['val_default']:.2%}", f"{f['val_optimized']:.2%}"], textposition="outside"))
        fig.update_layout(height=340, yaxis=dict(range=[80, 96], title="Précision de validation (%)"), margin=dict(t=10, b=10))
        plot(fig)
        c = st.columns(2)
        c[0].metric("Test – par défaut", "92.92 %")
        c[1].metric("Test – optimisée", "88.93 %", "-3.99 pts", delta_color="inverse")
        note(f"<b>Modèle retenu : {f['retained']}</b> (sélection faite sur la validation, pas sur le test). "
             "La recherche reposait sur des essais de 3 époques : elle favorise les configurations qui progressent vite au début, "
             "pas celles qui profitent d'un entraînement long. Enfin, les écarts entre essais sont dans la zone de bruit "
             "(±1,5 pt). L'optimisation <i>confirme</i> donc que le choix raisonné (lr 1e-5, 30 couches) est bon plutôt qu'elle ne l'améliore.", "warn")
        curves(HIST["optimized"], "Fine-tuning avec la configuration optimisée")

    with t5:
        allr = pd.concat([grid.assign(méthode="Grid Search"), rnd.assign(méthode="Random Search")], ignore_index=True)
        c1, c2 = st.columns(2)
        meth = c1.multiselect("Méthode", ["Grid Search", "Random Search"], default=["Grid Search", "Random Search"])
        min_acc = c2.slider("Précision de validation minimale", 0.87, 0.90, 0.87, 0.001)
        view = allr[allr["méthode"].isin(meth) & (allr["val_acc"] >= min_acc)].sort_values(
            ["val_acc", "val_loss"], ascending=[False, True])
        table(view[["méthode", "lr", "n_layers", "dropout", "val_acc", "val_loss", "seconds"]].round(5))
        st.caption(f"{len(view)} essai(s) sur {len(allr)} · coût total ≈ {allr['seconds'].sum() / 60:.0f} min de GPU.")


def page_filters():
    header("Étape 8", "Filtres et cartes de caractéristiques",
           "Les premiers filtres détectent bords et couleurs ; plus profond, les cartes deviennent abstraites et moins résolues.")
    model, msg = get_model()
    t1, t2 = st.tabs(["Résultats du notebook", "Explorer en direct"])
    with t1:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            asset("filters", "16 premiers filtres 7×7 de la 1ʳᵉ convolution (shape 7×7×3×64)")
        with c2:
            asset("analysed_image", "Image analysée")
        asset("feature_maps", "8 cartes de caractéristiques à 4 profondeurs (conv 0, 5, 20, 40)")
        note("Premières couches : les cartes ressemblent encore à l'image (bords, contrastes) en haute résolution. "
             "Plus profond : la résolution diminue (stride, pooling) et les activations représentent des motifs plus abstraits.")
    with t2:
        if model is None:
            note(f"{msg}. Placez <code>{MODEL_NAME}</code> dans le dossier <code>models/</code> (ou chargez-le dans la barre latérale) "
                 "pour explorer les filtres et feature maps de n'importe quelle couche.", "warn")
            return
        import keras
        base = base_of(model)
        if base is None:
            st.error("Base ResNet introuvable dans le modèle.")
            return
        convs = [l for l in base.layers if isinstance(l, keras.layers.Conv2D)]
        st.caption(f"{len(convs)} couches de convolution dans ResNet50V2.")
        k = convs[0].get_weights()[0]
        n_f = st.slider("Nombre de filtres affichés (1ʳᵉ convolution)", 8, 64, 32, 8)
        fl = [(lambda a: (a - a.min()) / (a.max() - a.min() + 1e-8))(k[:, :, :, i]).astype("float32") for i in range(n_f)]
        show_image(tile(fl, 8), "Filtres de la première couche (normalisés pour l'affichage)")
        up = st.file_uploader("Image à analyser", type=["jpg", "jpeg", "png"], key="fm_up")
        img = Image.open(up) if up else sample_flower("daisy")
        idx = st.slider("Couche de convolution", 0, len(convs) - 1, 5)
        ext = keras.Model(base.input, convs[idx].output)
        x = (prepare_image(img)[None] / 127.5) - 1.0
        fm = ext.predict(x, verbose=0)[0]
        n_m = st.slider("Nombre de cartes", 4, min(48, fm.shape[-1]), min(16, fm.shape[-1]))
        c1, c2 = st.columns([1, 3])
        c1.image(img.resize((224, 224)), caption="Entrée")
        with c2:
            show_image(tile([viridis(fm[:, :, i]) for i in range(n_m)], 8),
                       f"{convs[idx].name} — {fm.shape[0]}×{fm.shape[1]}, {fm.shape[2]} canaux")


def page_evaluation():
    header("Étape 9", "Évaluation finale du modèle retenu",
           "Évalué sur les 551 images de test, restées inutilisées jusqu'ici.")
    rep = R["report"]
    c = st.columns(4)
    c[0].metric("Accuracy", f"{rep['accuracy']:.1%}")
    c[1].metric("Macro F1", f"{np.mean([v['f1'] for v in rep['per_class'].values()]):.3f}")
    c[2].metric("Erreurs", f"{rep['n_errors']} / {rep['n_test']}")
    c[3].metric("Test loss", "0.2252")
    t1, t2, t3, t4 = st.tabs(["Rapport par classe", "Matrice de confusion", "Analyse des erreurs", "Confiance"])
    with t1:
        pc = pd.DataFrame(rep["per_class"]).T
        pc.index.name = "classe"
        metric_sel = st.multiselect("Métriques", ["precision", "recall", "f1"], default=["precision", "recall", "f1"])
        fig = go.Figure()
        for m, col in zip(["precision", "recall", "f1"], [TEAL, GOLD, CORAL]):
            if m in metric_sel:
                fig.add_trace(go.Bar(name=m, x=pc.index, y=pc[m], marker_color=col, text=pc[m].round(3), textposition="outside"))
        fig.update_layout(barmode="group", height=360, yaxis=dict(range=[0.8, 1.0]), margin=dict(t=10, b=10))
        plot(fig)
        table(pc.reset_index().rename(columns={"classe": "Classe", "precision": "Précision", "recall": "Rappel",
                                               "f1": "F1", "support": "Support"}))
        worst = pc["f1"].idxmin(); best = pc["f1"].idxmax()
        note(f"Classe la plus facile : <b>{best}</b> (F1 {pc.loc[best, 'f1']:.3f}) · la plus difficile : <b>{worst}</b> (F1 {pc.loc[worst, 'f1']:.3f}).")
    with t2:
        asset("confusion_matrix", "Effectifs (gauche) et normalisée par classe réelle (droite)")
    with t3:
        asset("errors", "Quelques images mal classées (réel / prédit / confiance)")
        st.caption("À examiner pour repérer les causes : images ambiguës, plusieurs fleurs, vue éloignée, classes visuellement proches.")
    with t4:
        asset("confidence", "Probabilité maximale prédite : prédictions correctes vs erreurs")
        st.caption("Un modèle bien calibré est confiant quand il a raison et moins quand il se trompe.")


def page_conclusion():
    header("Étape 10", "Comparaison, sauvegarde et conclusion")
    df = pd.DataFrame(R["models"])
    t1, t2 = st.tabs(["Comparaison des approches", "Conclusion"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=df["name"], y=df["test_acc"], marker_color=MODEL_COLORS,
                                   text=[f"{v:.2f}" for v in df["test_acc"]], textposition="outside"))
            fig.update_layout(title="Accuracy test (%)", yaxis=dict(range=[0, 100]), height=380, margin=dict(t=50, b=10))
            fig.update_xaxes(tickangle=-20)
            plot(fig)
        with c2:
            fig = go.Figure(go.Bar(x=df["name"], y=df["test_loss"], marker_color=MODEL_COLORS,
                                   text=[f"{v:.4f}" for v in df["test_loss"]], textposition="outside"))
            fig.update_layout(title="Test loss (cross-entropy)", height=380, margin=dict(t=50, b=10))
            fig.update_xaxes(tickangle=-20)
            plot(fig)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Entraînables", x=df["name"], y=df["trainable"], marker_color=CORAL))
        fig.add_trace(go.Bar(name="Figés", x=df["name"], y=df["frozen"], marker_color=GREY))
        fig.update_layout(barmode="stack", title="Paramètres entraînables vs figés", height=340, margin=dict(t=50, b=10))
        plot(fig)
        out = df.rename(columns={"name": "Modèle", "test_loss": "Test loss", "test_acc": "Test accuracy (%)",
                                 "trainable": "Entraînables", "frozen": "Figés"})
        table(out)
    with t2:
        st.markdown("""
        **Ce que l'on retient**
        1. Le **transfert d'apprentissage** apporte le gain principal (69,9 % → 88,0 %) : les caractéristiques ImageNet sont réutilisables sur un petit jeu de données.
        2. Le **fine-tuning** ajoute +4,9 pts (92,9 %), à condition d'utiliser un learning rate très faible et de garder les BatchNorm figées.
        3. L'**optimisation des hyperparamètres** vérifie les choix initiaux ; ici elle ne les améliore pas (essais courts, écarts dans le bruit).
        4. **Régularisation** (L2, Dropout, augmentation), **early stopping** et **learning rate adaptatif** limitent le sur-apprentissage.
        5. Sélection sur la **validation**, test réservé à l'évaluation finale.
        6. L'évaluation va au-delà de l'accuracy : rapport par classe, matrice de confusion, analyse des erreurs.
        """)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Limites**")
            st.markdown("- Un seul découpage train / validation / test et des essais de 3 époques.\n"
                        "- Espace de recherche restreint (3 hyperparamètres, 15 essais).\n"
                        "- Jeu de données modeste, avec quelques images ambiguës.")
        with c2:
            st.markdown("**Perspectives**")
            st.markdown("- Optimisation bayésienne (Keras Tuner, Optuna).\n"
                        "- Autres hyperparamètres : λ, batch size, optimiseur, intensité d'augmentation.\n"
                        "- Autres architectures : EfficientNet, MobileNetV2, DenseNet, Vision Transformers.")
        st.info(f"Modèle sauvegardé dans le notebook : `{MODEL_NAME}`")


def page_inference():
    header("Étape 11", "Inférence : testez le modèle",
           "Chargez une photo de fleur ; le modèle fine-tuné prédit sa classe.")
    model, msg = get_model()
    if model is None:
        note(f"<b>{msg}.</b> Cette page a besoin du modèle entraîné : dans Colab, téléchargez "
             f"<code>{MODEL_NAME}</code> (panneau Fichiers), puis placez-le dans <code>models/</code> ou chargez-le "
             "dans la barre latérale. TensorFlow ≥ 2.16 / Keras 3 est requis.", "warn")
        st.subheader("Exemple du notebook")
        asset("inference_example", "Image externe de tournesol : sunflowers 100.0 %, dandelion 0.0 %, daisy 0.0 %")
        return
    c1, c2 = st.columns([1, 1.2])
    with c1:
        mode = st.radio("Source", ["Téléverser", "Caméra", "Dessin d'exemple"], horizontal=True)
        file = None
        if mode == "Téléverser":
            file = st.file_uploader("Photo de fleur", type=["jpg", "jpeg", "png"])
        elif mode == "Caméra":
            file = st.camera_input("Prendre une photo")
        img = Image.open(file) if file else (sample_flower("daisy") if mode == "Dessin d'exemple" else None)
        top_k = st.slider("Top-k", 1, 5, 3)
        if img is not None:
            show_image(img, "Image analysée")
    with c2:
        if img is None:
            st.info("Ajoutez une image pour lancer la prédiction.")
            return
        proba = predict(model, img)
        order = np.argsort(proba)[::-1]
        best = int(order[0])
        st.metric("Prédiction", f"{CLASSES_FR[CLASSES[best]]} ({CLASSES[best]})", f"{proba[best]:.1%} de confiance", delta_color="off")
        fig = go.Figure(go.Bar(x=[proba[i] for i in order[:top_k]][::-1],
                               y=[CLASSES[i] for i in order[:top_k]][::-1], orientation="h",
                               marker_color=[CLASS_COLORS[CLASSES[i]] for i in order[:top_k]][::-1],
                               text=[f"{proba[i]:.1%}" for i in order[:top_k]][::-1], textposition="outside"))
        fig.update_layout(height=60 + 60 * top_k, xaxis=dict(range=[0, 1.15], title="Probabilité"), margin=dict(t=10, b=10))
        plot(fig)
        if proba[best] < 0.6:
            note("Confiance faible : l'image est peut-être ambiguë ou ne représente pas l'une des 5 classes apprises.", "warn")


# --------------------------------------------------------------------------- navigation
PAGES = {
    "🏠 Vue d'ensemble": page_home,
    "1 · Configuration": page_config,
    "2 · Données": page_data,
    "3 · Data augmentation": page_augmentation,
    "4 · CNN from scratch": page_baseline,
    "5 · Transfert (base gelée)": page_transfer,
    "6 · Fine-tuning": page_finetune,
    "7 · Optimisation": page_optimization,
    "8 · Filtres & feature maps": page_filters,
    "9 · Évaluation": page_evaluation,
    "10 · Conclusion": page_conclusion,
    "11 · Inférence": page_inference,
}
NAMES = list(PAGES)
if "page" not in st.session_state:
    st.session_state["page"] = NAMES[0]


def go_to(delta: int):
    i = NAMES.index(st.session_state["page"]) + delta
    st.session_state["page"] = NAMES[max(0, min(len(NAMES) - 1, i))]


with st.sidebar:
    st.markdown("### 🌼 Projet Deep Learning")
    st.caption("Fine-tuning ResNet50V2 · tf_flowers")
    st.radio("Étapes", NAMES, key="page", label_visibility="collapsed")
    idx = NAMES.index(st.session_state["page"])
    st.progress(idx / (len(NAMES) - 1), text=f"Étape {idx} / {len(NAMES) - 1}")
    st.divider()
    up_model = st.file_uploader("Charger le modèle (.keras)", type=["keras"], help="Pour l'inférence et l'exploration des filtres")
    if up_model is not None:
        MODELS_DIR.mkdir(exist_ok=True)
        dest = MODELS_DIR / MODEL_NAME
        if not dest.exists() or dest.stat().st_size != up_model.size:
            dest.write_bytes(up_model.getbuffer())
            st.cache_resource.clear()
    st.caption("Modèle : " + ("✅ " + MODEL_NAME if any(MODELS_DIR.glob("*.keras")) else "non chargé (mode figures du notebook)"))

PAGES[st.session_state["page"]]()

st.divider()
c1, _, c3 = st.columns([1, 4, 1])
idx = NAMES.index(st.session_state["page"])
if idx > 0:
    c1.button("← Précédent", on_click=go_to, args=(-1,))
if idx < len(NAMES) - 1:
    c3.button("Suivant →", on_click=go_to, args=(1,))
