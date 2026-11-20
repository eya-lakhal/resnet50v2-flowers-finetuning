"""Extrait du notebook Colab exécuté : courbes d'entraînement, résultats d'optimisation, métriques
et figures. Génère data/results.json et assets/*.png utilisés par l'application Streamlit.

Usage : python extract_data.py chemin/vers/Projet_FineTuning.ipynb
"""
import base64, json, re, sys
from pathlib import Path

nb_path = Path(sys.argv[1])
nb = json.load(open(nb_path, encoding="utf-8"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def out_text(i):
    parts = []
    for o in nb["cells"][i].get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain") or []
        parts.append("".join(t))
    return ANSI.sub("", "".join(parts))


def parse_history(i):
    rows = []
    pat = re.compile(r"accuracy: ([\d.]+) - loss: ([\d.]+) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+) - learning_rate: ([\d.e+-]+)")
    for m in pat.finditer(out_text(i)):
        rows.append(dict(acc=float(m[1]), loss=float(m[2]), val_acc=float(m[3]),
                         val_loss=float(m[4]), lr=float(m[5])))
    return rows


def parse_trials(i, random=False):
    rows = []
    pat = re.compile(r"\[(\d+)/\d+\] lr=([\d.e+-]+) \| couches(?: dégelées)?=\s*(\d+)"
                     + (r" \| dropout=([\d.]+)" if random else "")
                     + r" \| val_acc=([\d.]+) \| val_loss=([\d.]+) \| (\d+)s")
    for m in pat.finditer(out_text(i)):
        g = m.groups()
        if random:
            rows.append(dict(lr=float(g[1]), n_layers=int(g[2]), dropout=float(g[3]),
                             val_acc=float(g[4]), val_loss=float(g[5]), seconds=int(g[6])))
        else:
            rows.append(dict(lr=float(g[1]), n_layers=int(g[2]), dropout=0.3,
                             val_acc=float(g[3]), val_loss=float(g[4]), seconds=int(g[5])))
    return rows


def find_cell(snippet, start=0):
    for i in range(start, len(nb["cells"])):
        if nb["cells"][i]["cell_type"] == "code" and snippet in "".join(nb["cells"][i]["source"]):
            return i
    raise KeyError(snippet)


c_base = find_cell("history_baseline = baseline.fit")
c_head = find_cell("history_head = model.fit")
c_ft = find_cell("history_ft = model.fit")
c_opt = find_cell("history_opt = model.fit")
c_grid = find_cell("grid_records = []")
c_rand = find_cell("random_records = []")
c_cv = find_cell("GridSearchCV(")
c_cv2 = find_cell("RandomizedSearchCV(clf")
c_rep = find_cell("classification_report(y_true")
c_res = find_cell("df_results = pd.DataFrame(results)")

res = {"histories": {
    "baseline": parse_history(c_base), "phase1": parse_history(c_head),
    "finetune": parse_history(c_ft), "optimized": parse_history(c_opt)}}
res["grid"] = parse_trials(c_grid)
res["random"] = parse_trials(c_rand, random=True)

# --- validation croisée sur la tête (SciKeras / wrapper sklearn) ---
t = out_text(c_cv); t2 = out_text(c_cv2)
res["cv"] = {
    "default_head_val": float(re.search(r"Tête par défaut -> précision de validation = ([\d.]+)", t)[1]),
    "grid": {"params": eval(re.search(r"GridSearchCV -> meilleurs hyperparamètres : (\{.*?\})", t)[1]),
             "cv_score": float(re.search(r"Score de validation croisée : ([\d.]+)", t)[1]),
             "val_score": float(re.search(r"Précision sur la validation : ([\d.]+)", t)[1]),
             "n_fits": 27},
    "random": {"params": eval(re.search(r"RandomizedSearchCV -> meilleurs hyperparamètres : (\{.*?\})", t2)[1]),
               "cv_score": float(re.search(r"Score de validation croisée : ([\d.]+)", t2)[1]),
               "val_score": float(re.search(r"Précision sur la validation : ([\d.]+)", t2)[1]),
               "n_fits": 30},
    # top-5 affiché dans le notebook (mean, std)
    "random_top5": [
        {"rank": 1, "lr": 1e-3, "l2": 1e-5, "mean": 0.875825, "std": 0.006388},
        {"rank": 2, "lr": 3e-4, "l2": 1e-3, "mean": 0.874271, "std": 0.006756},
        {"rank": 3, "lr": 3e-4, "l2": 1e-2, "mean": 0.872712, "std": 0.008755},
        {"rank": 4, "lr": 3e-4, "l2": 1e-5, "mean": 0.852862, "std": 0.009373},
        {"rank": 5, "lr": 1e-2, "l2": 1e-2, "mean": 0.851694, "std": 0.012898}],
}

# --- rapport de classification ---
rep = out_text(c_rep)
per_class = {}
for m in re.finditer(r"^\s*(daisy|dandelion|roses|sunflowers|tulips)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", rep, re.M):
    per_class[m[1]] = dict(precision=float(m[2]), recall=float(m[3]), f1=float(m[4]), support=int(m[5]))
res["report"] = {"per_class": per_class, "accuracy": float(re.search(r"accuracy\s+([\d.]+)\s+\d+", rep)[1]),
                 "n_test": 551, "n_errors": 39}

# --- tableau comparatif ---
rt = out_text(c_res)
models = []
for name in ["CNN from scratch", "ResNet50V2 : base gelée", "ResNet50V2 : fine-tuning", "ResNet50V2 : fine-tuning optimisé"]:
    m = re.search(re.escape(name) + r"\s+([\d.]+)\s+([\d.]+)", rt)
    p = re.search(re.escape(name) + r"\s+([\d.]+)\s+([\d.]+)\s*$", rt[rt.index("Paramètres"):], re.M)
    models.append(dict(name=name, test_loss=float(m[1]), test_acc=float(m[2])))
params = re.findall(r"^(CNN from scratch|ResNet50V2 : [^\n]*?)\s{2,}([\d.]+)\s+([\d.]+)\s*$", rt[rt.index("Paramètres"):], re.M)
for mod, (_, tr, fr) in zip(models, params):
    mod["trainable"] = int(float(tr)); mod["frozen"] = int(float(fr))
for mod in models:                       # ligne éventuellement tronquée dans la sortie : valeurs lues dans le notebook
    if "trainable" not in mod:
        mod["trainable"], mod["frozen"] = 3420165, 20154880
res["models"] = models
res["data"] = {"classes": ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
               "train": [445, 633, 455, 480, 556], "val": [104, 126, 96, 104, 120], "test": [84, 139, 90, 115, 123]}
res["final"] = {"val_default": 0.9255, "val_optimized": 0.8873,
                "best_cfg": {"lr": 1.2953707094821835e-05, "n_layers": 10, "dropout": 0.3931595360241994},
                "retained": "configuration par défaut"}

Path("data").mkdir(exist_ok=True)
json.dump(res, open("data/results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# --- figures ---
names = {"c6_0": "activations", "c10_0": "gradient_descent", "c15_0": "split_distribution",
         "c16_0": "samples", "c20_0": "augmentation", "c26_0": "curves_baseline",
         "c34_0": "curves_phase1", "c39_0": "curves_finetune", "c45_0": "grid_heatmap",
         "c49_0": "random_scatter", "c59_0": "curves_optimized", "c62_0": "filters",
         "c63_0": "analysed_image", "c63_1": "feature_maps", "c67_0": "confusion_matrix",
         "c68_0": "errors", "c69_0": "confidence", "c72_0": "comparison", "c74_0": "inference_example"}
Path("assets").mkdir(exist_ok=True)
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    n = 0
    for o in c.get("outputs", []):
        png = o.get("data", {}).get("image/png")
        if png:
            key = f"c{i}_{n}"; n += 1
            if key in names:
                Path(f"assets/{names[key]}.png").write_bytes(base64.b64decode(png))
print("OK:", {k: len(v) for k, v in res["histories"].items()}, len(res["grid"]), len(res["random"]))
