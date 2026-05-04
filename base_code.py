"""
╔═════════════════════════════════════════════════════════════════════════════╗
║   ANTENNA BANDWIDTH — ACTIVE LEARNING PIPELINE                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║  PURPOSE:                                                                   ║
║    Train on 80% simulation data → identify where model is uncertain →       ║
║    YOU run those uncertain combos in simulator → feed results back →        ║
║    Retrain → model gets better at REAL antenna physics, not just fitting    ║
║                                                                             ║
║  WORKFLOW (repeat each round):                                              ║
║    Step 1: python active_learning.py --round 0   (first training)           ║
║    Step 2: Look at suggested_round_0.csv          (combos to simulate)      ║
║    Step 3: Run those in your simulation software                            ║
║    Step 4: Fill results into new_data_round_0.csv                           ║
║    Step 5: python active_learning.py --round 1   (retrain with new data)    ║
║    Step 6: Repeat until accuracy satisfies you                              ║
║                                                                             ║
║  WHY THIS WORKS:                                                            ║
║    • Model picks combos where it is MOST UNCERTAIN (not random)             ║
║    • Each new simulation point gives maximum information gain               ║
║    • 10-20 targeted simulations can improve accuracy more than              ║
║      100 random ones                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os

# Keep matplotlib/font caches inside writable temp storage.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wideband-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/wideband-cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, pickle, json, argparse
from copy import deepcopy
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                               RandomForestClassifier, RandomForestRegressor,
                               HistGradientBoostingClassifier,
                               HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report,
                              r2_score, mean_absolute_error, balanced_accuracy_score,
                              log_loss,
                              matthews_corrcoef)

# ─────────────── CONFIG ──────────────────────────────────────────
DATA_PATH      = 'UBW-data-final.xlsx'   # your original simulation data
HISTORY_FILE   = 'active_learning_history.json'
OUTPUT_DIR     = '.'
N_SUGGEST      = 20    # how many new combos to suggest per round
TEST_SIZE      = 0.20
EPOCHS         = 20

# All valid parameter values from your data
LG_VALS = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
WR_VALS = [0.75, 2.25, 2.5, 7.5, 8.5, 9.25, 9.5, 10.0, 10.5, 10.75,
            11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75]
WF_VALS = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (52 features — same as model_v3)
# ══════════════════════════════════════════════════════════════════
WR_B1=8.5; WR_B2=10.75; WR_B3=11.0; WR_B4=11.25; WR_B5=12.0
WF_PERIOD=0.55; LG_PERIOD=4.25

def build_features(df_in):
    if isinstance(df_in, pd.DataFrame):
        LG = df_in['LG'].values.astype(float)
        WR = df_in['WR'].values.astype(float)
        WF = df_in['WF'].values.astype(float)
    else:
        LG, WR, WF = df_in[:,0], df_in[:,1], df_in[:,2]
    f = {}
    f['LG']=LG; f['WR']=WR; f['WF']=WF
    f['LG2']=LG**2; f['WR2']=WR**2; f['WF2']=WF**2
    f['LG3']=LG**3; f['WR3']=WR**3
    f['LG_WR']=LG*WR; f['LG_WF']=LG*WF; f['WR_WF']=WR*WF
    f['LG_WR_WF']=LG*WR*WF; f['LG2_WR']=LG**2*WR
    f['WR_b1']=(WR>WR_B1).astype(float); f['WR_b2']=(WR>WR_B2).astype(float)
    f['WR_b3']=(WR>WR_B3).astype(float); f['WR_b4']=(WR>WR_B4).astype(float)
    f['WR_b5']=(WR>WR_B5).astype(float)
    f['WR_d11']=WR-11.0; f['WR_d1125']=np.clip(WR-11.25,0,None)
    f['WR_zone']=np.where(WR<WR_B1,0,np.where(WR<WR_B2,1,np.where(WR<WR_B3,2,
                  np.where(WR<WR_B4,3,np.where(WR<WR_B5,4,5))))).astype(float)
    f['zone_LG']=f['WR_zone']*LG
    for k in [1,2,3]:
        ang_wf=k*np.pi*WF/WF_PERIOD
        f[f'sinWF{k}']=np.sin(ang_wf); f[f'cosWF{k}']=np.cos(ang_wf)
    for k in [1,2]:
        ang_lg=k*np.pi*LG/LG_PERIOD
        f[f'sinLG{k}']=np.sin(ang_lg); f[f'cosLG{k}']=np.cos(ang_lg)
    f['sinWF1_LG']=np.sin(np.pi*WF/WF_PERIOD)*LG
    f['cosWF1_LG']=np.cos(np.pi*WF/WF_PERIOD)*LG
    f['logWR']=np.log1p(WR); f['logLG']=np.log(LG); f['logWF']=np.log(WF)
    f['WR_WFr']=WR/(WF+0.1); f['LG_WRr']=LG/(WR+0.1); f['WR_LGr']=WR/(LG+0.1)
    for bnd in [8.5,10.75,11.0,11.25,12.0,12.5]: f[f'dWR_{bnd}']=WR-bnd
    f['WR_b3_LG_WF']=f['WR_b3']*LG*WF; f['WR_d11_LG']=f['WR_d11']*LG
    f['WR_d11_WF']=f['WR_d11']*WF; f['WF2_LG']=WF**2*LG
    f['WR2_LG']=WR**2*LG; f['LG_WF_zone']=LG*WF*f['WR_zone']
    return pd.DataFrame(f)


# ══════════════════════════════════════════════════════════════════
# LOAD HISTORY
# ══════════════════════════════════════════════════════════════════
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE,'r') as f:
            return json.load(f)
    return {'rounds':[], 'all_new_data':[], 'metrics_per_round':[]}

def save_history(h):
    with open(HISTORY_FILE,'w') as f:
        json.dump(h, f, indent=2)


# ══════════════════════════════════════════════════════════════════
# LOAD ALL DATA (original + any previously collected)
# ══════════════════════════════════════════════════════════════════
def load_all_data(history):
    # Original simulation data
    df_orig = pd.read_excel(DATA_PATH, header=0, usecols=[0,1,2,3,4,5,6])
    df_orig.columns = ['SrNo','LG','WR','WF','FL','FU','BW']
    df_orig = df_orig.dropna(subset=['LG','WR','WF','BW']).reset_index(drop=True)
    df_orig = (df_orig.sort_values('BW', ascending=False)
                      .drop_duplicates(subset=['LG','WR','WF'], keep='first')
                      .reset_index(drop=True))
    df_orig['source'] = 'original'

    frames = [df_orig]

    # Any new data collected in previous rounds
    for r_idx, round_data in enumerate(history.get('all_new_data', [])):
        if round_data:
            df_new = pd.DataFrame(round_data)
            df_new['source'] = f'round_{r_idx}'
            frames.append(df_new)

    df_all = pd.concat(frames, ignore_index=True)
    # Final dedup
    df_all = (df_all.sort_values(['source','BW'], ascending=[True, False])
                    .drop_duplicates(subset=['LG','WR','WF'], keep='first')
                    .reset_index(drop=True))
    return df_all


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)


def safe_classification_report(y_true, y_pred):
    labels = [0, 1]
    target_names = ['Dead', 'Resonant']
    present = set(np.unique(y_true).tolist())
    if present.issubset({0, 1}) and len(present) == 1:
        # Keep the report readable when the validation split has one class only.
        return classification_report(
            y_true, y_pred, labels=labels, target_names=target_names,
            zero_division=0
        )
    return classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        zero_division=0
    )


def make_safe_test_split(df_all, test_size=TEST_SIZE, seed=42):
    """
    Create a holdout split that keeps the test set close to 10% of the data
    while ensuring the rare dead class appears in the test set whenever the
    dataset has enough samples to support it.
    """
    y = (df_all.BW > 0).astype(int).values
    idx_all = np.arange(len(df_all))
    dead_idx = idx_all[y == 0]
    res_idx = idx_all[y == 1]

    n_total = len(df_all)
    n_test = max(1, int(round(test_size * n_total)))
    rng = np.random.default_rng(seed)

    # If we have at least two dead samples, hold out one dead sample so the
    # test metrics can actually measure the minority class.
    if len(dead_idx) >= 2 and len(res_idx) >= 2:
        n_dead_test = max(1, int(round(test_size * len(dead_idx))))
        n_dead_test = min(n_dead_test, len(dead_idx) - 1)

        n_res_test = n_test - n_dead_test
        n_res_test = max(1, min(n_res_test, len(res_idx) - 1))

        # Adjust if rounding drift changes the final test size.
        while n_dead_test + n_res_test > n_test and n_res_test > 1:
            n_res_test -= 1
        while n_dead_test + n_res_test < n_test and n_res_test < len(res_idx) - 1:
            n_res_test += 1

        test_dead = rng.choice(dead_idx, size=n_dead_test, replace=False)
        test_res = rng.choice(res_idx, size=n_res_test, replace=False)
        test_idx = np.sort(np.concatenate([test_dead, test_res]))
        train_idx = np.setdiff1d(idx_all, test_idx, assume_unique=False)

        return (
            df_all.iloc[train_idx].reset_index(drop=True),
            df_all.iloc[test_idx].reset_index(drop=True),
            {
                'train_dead': int((df_all.iloc[train_idx].BW > 0).eq(0).sum()),
                'train_res': int((df_all.iloc[train_idx].BW > 0).sum()),
                'test_dead': int((df_all.iloc[test_idx].BW > 0).eq(0).sum()),
                'test_res': int((df_all.iloc[test_idx].BW > 0).sum()),
            }
        )

    # Fallback: if the dataset is too small/imbalanced to enforce the above,
    # use a deterministic random split and report what happened.
    shuffled = rng.permutation(idx_all)
    test_idx = np.sort(shuffled[:n_test])
    train_idx = np.sort(shuffled[n_test:])
    return (
        df_all.iloc[train_idx].reset_index(drop=True),
        df_all.iloc[test_idx].reset_index(drop=True),
        {
            'train_dead': int((df_all.iloc[train_idx].BW > 0).eq(0).sum()),
            'train_res': int((df_all.iloc[train_idx].BW > 0).sum()),
            'test_dead': int((df_all.iloc[test_idx].BW > 0).eq(0).sum()),
            'test_res': int((df_all.iloc[test_idx].BW > 0).sum()),
        }
    )


def _save_monitor_plot(fig, path, dark):
    fig.savefig(path, facecolor=dark, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return path


def make_epoch_monitor_figures(round_idx, epochs, train_accs, test_accs, train_losses,
                               y_test, y_test_pred):
    bg = "#ffffff"
    panel = "#f7f7f7"
    border = "#d0d0d0"
    text = "#111111"

    def dark_ax(ax):
        ax.set_facecolor(panel)
        for s in ax.spines.values():
            s.set_edgecolor(border)
        ax.tick_params(colors=text, labelsize=8)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.title.set_color(text)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=bg)
    dark_ax(ax)
    ax.plot(epochs, train_losses, color="#ffcc00", lw=2, marker="o", ms=4)
    ax.scatter(epochs, train_losses, color="#111111", s=18, zorder=3)
    for e, v in zip(epochs, train_losses):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6, color=text)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss - Round {round_idx}")
    ax.set_xlim(1, max(epochs))
    ax.set_xticks(epochs)
    loss_path = os.path.join(OUTPUT_DIR, f'loss_round_{round_idx}.png')
    _save_monitor_plot(fig, loss_path, bg)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=bg)
    dark_ax(ax)
    ax.plot(epochs, train_accs, color="#00d4ff", lw=2, marker="o", ms=4, label="Train")
    ax.plot(epochs, test_accs, color="#00e87a", lw=2, marker="s", ms=4, label="Test")
    ax.scatter(epochs, train_accs, color="#111111", s=18, zorder=3)
    ax.scatter(epochs, test_accs, color="#444444", s=18, zorder=3)
    for e, v in zip(epochs, train_accs):
        ax.annotate(f"{v:.2f}", (e, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6, color=text)
    for e, v in zip(epochs, test_accs):
        ax.annotate(f"{v:.2f}", (e, v), textcoords="offset points", xytext=(0, -10),
                    ha="center", fontsize=6, color=text)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Accuracy Across Epochs - Round {round_idx}")
    ax.set_xlim(1, max(epochs))
    ax.set_xticks(epochs)
    ax.legend(fontsize=7, facecolor=panel, labelcolor="black")
    acc_path = os.path.join(OUTPUT_DIR, f'accuracy_round_{round_idx}.png')
    _save_monitor_plot(fig, acc_path, bg)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=bg)
    dark_ax(ax)
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    import seaborn as sns
    sns.heatmap(cm, ax=ax, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Dead", "Resonant"], yticklabels=["Dead", "Resonant"],
                annot_kws={"size": 14, "fontweight": "bold"}, linewidths=1)
    ax.set_title(f"Test Confusion Matrix - Round {round_idx}")
    [t.set_color("black") for t in ax.get_xticklabels() + ax.get_yticklabels()]
    cm_path = os.path.join(OUTPUT_DIR, f'confusion_round_{round_idx}.png')
    _save_monitor_plot(fig, cm_path, bg)

    return {'loss': loss_path, 'accuracy': acc_path, 'confusion': cm_path}


def make_epoch_history_figure(round_idx, epochs, train_accs, test_accs, train_losses):
    """
    Combined epoch history plot with loss and accuracy in one white-background figure.
    """
    bg = "#ffffff"
    panel = "#f7f7f7"
    border = "#d0d0d0"
    text = "#111111"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=bg)

    def style_ax(ax):
        ax.set_facecolor(panel)
        for s in ax.spines.values():
            s.set_edgecolor(border)
        ax.tick_params(colors=text, labelsize=8)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.title.set_color(text)

    ax = axes[0]
    style_ax(ax)
    ax.plot(epochs, train_losses, color="#f5b800", lw=2.2, marker="o", ms=5)
    ax.scatter(epochs, train_losses, color="#111111", s=20, zorder=3)
    for e, v in zip(epochs, train_losses):
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.5, color=text)
    ax.set_title(f"Epoch Loss History - Round {round_idx}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, max(epochs))
    ax.set_xticks(epochs)

    ax = axes[1]
    style_ax(ax)
    ax.plot(epochs, train_accs, color="#0066cc", lw=2.2, marker="o", ms=5, label="Train")
    ax.plot(epochs, test_accs, color="#008a3d", lw=2.2, marker="s", ms=5, label="Test")
    ax.scatter(epochs, train_accs, color="#111111", s=16, zorder=3)
    ax.scatter(epochs, test_accs, color="#444444", s=16, zorder=3)
    for e, v in zip(epochs, train_accs):
        ax.annotate(f"{v:.2f}", (e, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.5, color=text)
    for e, v in zip(epochs, test_accs):
        ax.annotate(f"{v:.2f}", (e, v), textcoords="offset points", xytext=(0, -11),
                    ha="center", fontsize=6.5, color=text)
    ax.set_title(f"Epoch Accuracy History - Round {round_idx}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, max(epochs))
    ax.set_xticks(epochs)
    ax.legend(fontsize=7, facecolor=panel, labelcolor="black")

    fig_path = os.path.join(OUTPUT_DIR, f'epoch_history_round_{round_idx}.png')
    fig.savefig(fig_path, facecolor=bg, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return fig_path


def train_epoch_monitor(df_train, df_test, round_idx):
    """
    Auxiliary epoch-based monitor used only for the requested 20-epoch plots.
    It does not replace the ensemble active-learning model.
    """
    X_train = build_features(df_train)
    y_train = (df_train.BW > 0).astype(int).values
    X_test = build_features(df_test)
    y_test = (df_test.BW > 0).astype(int).values

    monitor = Pipeline([
        ('sc', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.005,
            max_iter=1,
            warm_start=True,
            early_stopping=False,
            random_state=42,
        ))
    ])

    train_accs, test_accs, train_losses = [], [], []
    for _ in range(EPOCHS):
        monitor.fit(X_train, y_train)
        train_pred = monitor.predict(X_train)
        test_pred = monitor.predict(X_test)
        train_accs.append(accuracy_score(y_train, train_pred))
        test_accs.append(accuracy_score(y_test, test_pred))
        train_losses.append(float(monitor.named_steps['clf'].loss_))

    epoch_history = pd.DataFrame({
        'epoch': list(range(1, EPOCHS + 1)),
        'train_accuracy': train_accs,
        'test_accuracy': test_accs,
        'train_loss': train_losses,
    })
    epoch_csv = os.path.join(OUTPUT_DIR, f'epoch_history_round_{round_idx}.csv')
    epoch_history.to_csv(epoch_csv, index=False)

    final_test_pred = monitor.predict(X_test)
    final_test_acc = accuracy_score(y_test, final_test_pred)
    final_test_cm = confusion_matrix(y_test, final_test_pred, labels=[0, 1])
    final_test_report = safe_classification_report(y_test, final_test_pred)

    fig_paths = make_epoch_monitor_figures(
        round_idx, epochs=list(range(1, EPOCHS + 1)),
        train_accs=train_accs, test_accs=test_accs, train_losses=train_losses,
        y_test=y_test, y_test_pred=final_test_pred
    )

    return {
        'model': monitor,
        'test_acc': float(final_test_acc),
        'test_cm': final_test_cm.tolist(),
        'test_report': final_test_report,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'fig_paths': fig_paths,
        'epoch_csv': epoch_csv,
        'epoch_history_fig': make_epoch_history_figure(
            round_idx, list(range(1, EPOCHS + 1)), train_accs, test_accs, train_losses
        ),
    }


def train_gradual_learning_curve(df_train, df_test, round_idx, n_steps=20):
    """
    Show how the monitor improves as it sees more of the training data.
    This is a true first-to-last-step learning curve, not an epoch curve.
    """
    bg = "#ffffff"
    panel = "#f7f7f7"
    border = "#d0d0d0"
    text = "#111111"

    X_train = build_features(df_train)
    y_train = (df_train.BW > 0).astype(int).values
    X_test = build_features(df_test)
    y_test = (df_test.BW > 0).astype(int).values

    dead_idx = np.where(y_train == 0)[0]
    res_idx = np.where(y_train == 1)[0]
    rng = np.random.default_rng(42)
    ordered_idx = np.concatenate([
        rng.permutation(dead_idx),
        rng.permutation(res_idx),
    ])

    fractions = np.linspace(0.1, 1.0, n_steps)
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    train_bal_accs = []
    test_bal_accs = []

    for frac in fractions:
        n_use = max(5, int(np.ceil(len(df_train) * frac)))
        subset_idx = ordered_idx[:min(n_use, len(ordered_idx))]
        X_sub = X_train.iloc[subset_idx]
        y_sub = y_train[subset_idx]

        # Ensure the subset is fit-able and contains both classes when possible.
        if len(np.unique(y_sub)) < 2 and len(res_idx) > 0 and len(dead_idx) > 0:
            extra = dead_idx[0] if 0 not in y_sub else res_idx[0]
            subset_idx = np.unique(np.append(subset_idx, extra))
            X_sub = X_train.iloc[subset_idx]
            y_sub = y_train[subset_idx]

        model = Pipeline([
            ('sc', QuantileTransformer(output_distribution='normal', random_state=42)),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=0.0005,
                alpha=0.003,
                max_iter=400,
                early_stopping=False,
                random_state=42,
            ))
        ])
        model.fit(X_sub, y_sub)

        train_prob = model.predict_proba(X_sub)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        train_pred = (train_prob >= 0.5).astype(int)
        test_pred = (test_prob >= 0.5).astype(int)

        train_accs.append(accuracy_score(y_sub, train_pred))
        test_accs.append(accuracy_score(y_test, test_pred))
        train_bal_accs.append(balanced_accuracy_score(y_sub, train_pred) if len(np.unique(y_sub)) > 1 else np.nan)
        test_bal_accs.append(balanced_accuracy_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else np.nan)
        train_losses.append(float(log_loss(y_sub, np.clip(train_prob, 1e-9, 1 - 1e-9), labels=[0, 1])))
        test_losses.append(float(log_loss(y_test, np.clip(test_prob, 1e-9, 1 - 1e-9), labels=[0, 1])))

    curve_df = pd.DataFrame({
        'fraction': fractions,
        'train_accuracy': train_accs,
        'test_accuracy': test_accs,
        'train_balanced_accuracy': train_bal_accs,
        'test_balanced_accuracy': test_bal_accs,
        'train_loss': train_losses,
        'test_loss': test_losses,
    })
    curve_csv = os.path.join(OUTPUT_DIR, f'gradual_learning_round_{round_idx}.csv')
    curve_df.to_csv(curve_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=bg)
    for ax in axes:
        ax.set_facecolor(panel)
        for s in ax.spines.values():
            s.set_edgecolor(border)
        ax.tick_params(colors=text, labelsize=8)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.title.set_color(text)

    ax = axes[0]
    ax.plot(fractions * 100, train_losses, color="#f5b800", lw=2.2, marker="o", ms=5, label="Train loss")
    ax.plot(fractions * 100, test_losses, color="#c2410c", lw=2.2, marker="s", ms=5, label="Test loss")
    for x, y in zip(fractions * 100, test_losses):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.5, color=text)
    ax.set_title(f"Gradual Learning Loss - Round {round_idx}")
    ax.set_xlabel("Training Data Used (%)")
    ax.set_ylabel("Log Loss")
    ax.set_xlim(10, 100)
    ax.legend(fontsize=7, facecolor=panel, labelcolor="black")

    ax = axes[1]
    ax.plot(fractions * 100, train_accs, color="#0066cc", lw=2.2, marker="o", ms=5, label="Train acc")
    ax.plot(fractions * 100, test_accs, color="#008a3d", lw=2.2, marker="s", ms=5, label="Test acc")
    for x, y in zip(fractions * 100, test_accs):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=6.5, color=text)
    ax.set_title(f"Gradual Learning Accuracy - Round {round_idx}")
    ax.set_xlabel("Training Data Used (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(10, 100)
    ax.legend(fontsize=7, facecolor=panel, labelcolor="black")

    fig_path = os.path.join(OUTPUT_DIR, f'gradual_learning_round_{round_idx}.png')
    fig.savefig(fig_path, facecolor=bg, dpi=110, bbox_inches='tight')
    plt.close(fig)

    return {
        'fig_path': fig_path,
        'csv_path': curve_csv,
        'fractions': fractions.tolist(),
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
    }


def evaluate_model_bundle(model_bundle, df_eval):
    """
    Evaluate a trained bundle on one holdout split.
    """
    X_eval = build_features(df_eval)
    y_eval = (df_eval.BW > 0).astype(int).values
    y_bw_eval = df_eval.BW.values

    base_proba_eval = np.column_stack([
        m.predict_proba(X_eval)[:, 1] for _, m in model_bundle['base_clfs']
    ])
    eval_prob = model_bundle['meta_lr'].predict_proba(base_proba_eval)[:, 1]
    eval_pred = (eval_prob >= model_bundle['best_thresh']).astype(int)
    eval_acc = accuracy_score(y_eval, eval_pred)
    eval_f1 = f1_score(y_eval, eval_pred, zero_division=0)
    eval_auc = safe_roc_auc(y_eval, eval_prob)
    eval_bacc = balanced_accuracy_score(y_eval, eval_pred)
    eval_cm = confusion_matrix(y_eval, eval_pred, labels=[0, 1])

    res_eval = (y_eval == 1)
    if res_eval.sum() > 0:
        X_res_eval = X_eval[res_eval]
        bw_pred_eval = np.dot(np.column_stack([
            model_bundle['et_bw'].predict(X_res_eval),
            model_bundle['rf_bw'].predict(X_res_eval),
            model_bundle['hgb_bw'].predict(X_res_eval)
        ]), model_bundle['REG_W'])
        eval_r2_bw = r2_score(y_bw_eval[res_eval], bw_pred_eval)
        eval_mae_bw = mean_absolute_error(y_bw_eval[res_eval], bw_pred_eval)
        eval_within01 = (np.abs(bw_pred_eval - y_bw_eval[res_eval]) < 0.1).mean() * 100
    else:
        eval_r2_bw = eval_mae_bw = eval_within01 = 0.0

    return {
        'acc': float(eval_acc),
        'f1': float(eval_f1),
        'auc': float(eval_auc),
        'bacc': float(eval_bacc),
        'cm': eval_cm.tolist(),
        'r2_bw': float(eval_r2_bw),
        'mae_bw': float(eval_mae_bw),
        'within01': float(eval_within01),
        'support_dead': int((y_eval == 0).sum()),
        'support_res': int((y_eval == 1).sum()),
        'y_eval': y_eval.tolist(),
        'eval_prob': eval_prob.tolist(),
        'eval_pred': eval_pred.tolist(),
    }


def make_resampling_summary_figure(round_idx, per_run_rows, bg="#ffffff"):
    panel = "#f7f7f7"
    border = "#d0d0d0"
    text = "#111111"
    metrics = [
        ('val_acc', 'Accuracy', '#0066cc'),
        ('val_bacc', 'Balanced Accuracy', '#008a3d'),
        ('val_f1', 'F1 Score', '#c2410c'),
        ('val_auc', 'AUC-ROC', '#7c3aed'),
        ('val_r2_bw', 'BW R²', '#b7791f'),
    ]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=bg)
    ax.set_facecolor(panel)
    for s in ax.spines.values():
        s.set_edgecolor(border)
    ax.tick_params(colors=text)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    ax.title.set_color(text)

    x = np.arange(len(metrics))
    means = [np.nanmean([r[k] for r in per_run_rows]) for k, _, _ in metrics]
    stds = [np.nanstd([r[k] for r in per_run_rows]) for k, _, _ in metrics]
    colors = [c for _, _, c in metrics]
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.9, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name, _ in metrics], rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f'Repeated-Split Evaluation Summary - Round {round_idx}')
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.02, f'{m:.3f}', ha='center', color=text, fontsize=8)

    fig_path = os.path.join(OUTPUT_DIR, f'resample_summary_round_{round_idx}.png')
    fig.savefig(fig_path, facecolor=bg, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return fig_path


def repeated_split_evaluation(df_all, round_idx, n_repeats=5, test_size=TEST_SIZE, seed=42):
    """
    Run repeated stratified holdout evaluations and average the metrics.
    """
    per_run = []
    for i in range(n_repeats):
        split_seed = seed + i
        df_train, df_val, split_info = make_safe_test_split(df_all, test_size=test_size, seed=split_seed)
        model_bundle, oof_metrics, _, _, _ = train_model(df_train)
        eval_metrics = evaluate_model_bundle(model_bundle, df_val)
        per_run.append({
            'repeat': i + 1,
            'seed': split_seed,
            'train_n': len(df_train),
            'val_n': len(df_val),
            'train_dead': split_info['train_dead'],
            'test_dead': split_info['test_dead'],
            'oof_acc': oof_metrics['oof_acc'],
            'oof_f1': oof_metrics['oof_f1'],
            'oof_auc': oof_metrics['oof_auc'],
            'oof_bacc': oof_metrics['oof_bacc'],
            'oof_r2_bw': oof_metrics['oof_r2_bw'],
            'oof_mae_bw': oof_metrics['oof_mae_bw'],
            'val_acc': eval_metrics['acc'],
            'val_f1': eval_metrics['f1'],
            'val_auc': eval_metrics['auc'],
            'val_bacc': eval_metrics['bacc'],
            'val_r2_bw': eval_metrics['r2_bw'],
            'val_mae_bw': eval_metrics['mae_bw'],
            'val_within01': eval_metrics['within01'],
        })

    summary = {}
    for key in ['val_acc', 'val_f1', 'val_auc', 'val_bacc', 'val_r2_bw', 'val_mae_bw', 'val_within01',
                'oof_acc', 'oof_f1', 'oof_auc', 'oof_bacc', 'oof_r2_bw', 'oof_mae_bw']:
        vals = np.array([r[key] for r in per_run], dtype=float)
        summary[key] = {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals))}

    summary['fig_path'] = make_resampling_summary_figure(round_idx=round_idx, per_run_rows=per_run)
    return per_run, summary


def make_technique_comparison_figure(round_idx, stage_metrics):
    """
    Show how the techniques progress from single models to combined voting/stacking.
    """
    bg = "#ffffff"
    panel = "#f7f7f7"
    border = "#d0d0d0"
    text = "#111111"

    stage_order = ['ET', 'RF', 'HGB', 'AVG2', 'AVG3', 'STACK']
    stage_labels = {
        'ET': 'ET',
        'RF': 'RF',
        'HGB': 'HGB',
        'AVG2': 'ET+RF',
        'AVG3': 'ET+RF+HGB',
        'STACK': 'Stack',
    }
    colors = {
        'ET': '#0066cc',
        'RF': '#008a3d',
        'HGB': '#c2410c',
        'AVG2': '#7c3aed',
        'AVG3': '#b7791f',
        'STACK': '#111111',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=bg)
    for ax in axes:
        ax.set_facecolor(panel)
        for s in ax.spines.values():
            s.set_edgecolor(border)
        ax.tick_params(colors=text, labelsize=8)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.title.set_color(text)

    metric_names = [('acc', 'Accuracy'), ('bacc', 'Balanced Accuracy'), ('f1', 'F1 Score'), ('auc', 'AUC-ROC')]
    x = np.arange(len(stage_order))

    ax = axes[0]
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(metric_names))
    for off, (m_key, m_label) in zip(offsets, metric_names):
        vals = [stage_metrics[s][m_key]['mean'] if s in stage_metrics else np.nan for s in stage_order]
        ax.bar(x + off, vals, width=width, label=m_label, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([stage_labels[s] for s in stage_order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f'Technique Comparison - Round {round_idx}')
    ax.legend(fontsize=7, facecolor=panel, labelcolor='black')

    ax = axes[1]
    acc_vals = [stage_metrics[s]['acc']['mean'] if s in stage_metrics else np.nan for s in stage_order]
    bacc_vals = [stage_metrics[s]['bacc']['mean'] if s in stage_metrics else np.nan for s in stage_order]
    ax.plot(stage_order, acc_vals, 'o-', color='#0066cc', lw=2, ms=6, label='Accuracy')
    ax.plot(stage_order, bacc_vals, 's-', color='#008a3d', lw=2, ms=6, label='Balanced Accuracy')
    for i, s in enumerate(stage_order):
        if s not in stage_metrics:
            continue
        ax.annotate(f"{acc_vals[i]:.2f}", (i, acc_vals[i]), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=7, color=text)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Stage')
    ax.set_ylabel('Score')
    ax.set_title('Learning Path From Single Models to Combined Ensemble')
    ax.legend(fontsize=7, facecolor=panel, labelcolor='black')

    fig_path = os.path.join(OUTPUT_DIR, f'technique_comparison_round_{round_idx}.png')
    fig.savefig(fig_path, facecolor=bg, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return fig_path


def make_technique_accuracy_loss_figures(round_idx, stage_metrics):
    """
    Save separate accuracy and loss plots for the technique ladder.
    """
    bg = "#ffffff"
    panel = "#f7f7f7"
    border = "#d0d0d0"
    text = "#111111"
    stage_order = ['ET', 'RF', 'HGB', 'AVG2', 'AVG3', 'STACK']
    stage_labels = ['ET', 'RF', 'HGB', 'ET+RF', 'ET+RF+HGB', 'Stack']
    colors = ['#0066cc', '#008a3d', '#c2410c', '#7c3aed', '#b7791f', '#111111']

    def style_ax(ax):
        ax.set_facecolor(panel)
        for s in ax.spines.values():
            s.set_edgecolor(border)
        ax.tick_params(colors=text, labelsize=8)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.title.set_color(text)

    acc_vals = [stage_metrics[s]['acc']['mean'] if s in stage_metrics else np.nan for s in stage_order]
    loss_vals = [stage_metrics[s]['loss']['mean'] if s in stage_metrics else np.nan for s in stage_order]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=bg)
    style_ax(ax)
    bars = ax.bar(stage_labels, acc_vals, color=colors, alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Technique Accuracy Progression - Round {round_idx}')
    for b, v in zip(bars, acc_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', color=text, fontsize=8)
    acc_path = os.path.join(OUTPUT_DIR, f'technique_accuracy_round_{round_idx}.png')
    fig.savefig(acc_path, facecolor=bg, dpi=110, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=bg)
    style_ax(ax)
    ax.plot(stage_labels, loss_vals, color="#f5b800", lw=2.2, marker="o", ms=6)
    ax.scatter(stage_labels, loss_vals, color="#111111", s=22, zorder=3)
    ax.set_ylabel('Log Loss')
    ax.set_title(f'Technique Loss Progression - Round {round_idx}')
    ax.set_ylim(0, max(loss_vals) * 1.25 if np.isfinite(np.nanmax(loss_vals)) else 1.0)
    for i, v in enumerate(loss_vals):
        ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(0, 6),
                    ha='center', color=text, fontsize=8)
    loss_path = os.path.join(OUTPUT_DIR, f'technique_loss_round_{round_idx}.png')
    fig.savefig(loss_path, facecolor=bg, dpi=110, bbox_inches='tight')
    plt.close(fig)

    return {'accuracy': acc_path, 'loss': loss_path}


# ══════════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════════
def train_model(df_train):
    """
    Train stacked ensemble on given dataframe.
    Returns model bundle and OOF CV metrics.
    """
    X = build_features(df_train)
    y_bin = (df_train.BW > 0).astype(int).values
    y_bw  = df_train.BW.values
    y_fl  = df_train.FL.values
    y_fu  = df_train.FU.values

    res_mask = (y_bin == 1)
    X_res    = X[res_mask]
    y_bw_res = y_bw[res_mask]
    y_fl_res = y_fl[res_mask]
    y_fu_res = y_fu[res_mask]

    class_counts = np.bincount(y_bin)
    min_class_count = class_counts[class_counts > 0].min()
    use_stacked_cv = min_class_count >= 2
    n_class_folds = max(2, min(5, int(min_class_count))) if use_stacked_cv else None
    skf = StratifiedKFold(n_splits=n_class_folds, shuffle=True, random_state=42) if use_stacked_cv else None
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    # Base classifiers
    et_clf  = ExtraTreesClassifier(n_estimators=500, max_depth=None,
                min_samples_leaf=1, max_features='sqrt',
                class_weight='balanced', random_state=42, n_jobs=-1)
    rf_clf  = RandomForestClassifier(n_estimators=300, max_depth=None,
                min_samples_leaf=2, max_features='sqrt',
                class_weight='balanced', random_state=42, n_jobs=-1)
    hgb_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                max_depth=6, class_weight='balanced', random_state=42)

    base_clfs = [('ET',et_clf),('RF',rf_clf),('HGB',hgb_clf)]

    if use_stacked_cv:
        # OOF predictions for meta-learner
        oof_proba = np.zeros((len(df_train), len(base_clfs)))
        base_oof = {nm: np.zeros(len(df_train)) for nm, _ in base_clfs}
        for fi, (ti, vi) in enumerate(skf.split(X, y_bin)):
            for mi, (nm, m) in enumerate(base_clfs):
                mc = deepcopy(m)
                mc.fit(X.iloc[ti], y_bin[ti])
                fold_prob = mc.predict_proba(X.iloc[vi])[:, 1]
                oof_proba[vi, mi] = fold_prob
                base_oof[nm][vi] = fold_prob

        meta_lr = LogisticRegressionCV(
            Cs=20, cv=n_class_folds, penalty='l2',
            class_weight='balanced', max_iter=2000, random_state=42
        )
        meta_lr.fit(oof_proba, y_bin)

        # Tune threshold
        best_thresh, best_f1 = 0.5, 0.0
        for t in np.arange(0.3, 0.71, 0.02):
            pred_t = (meta_lr.predict_proba(oof_proba)[:, 1] >= t).astype(int)
            f1_t = f1_score(y_bin, pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1, best_thresh = f1_t, t

        # Refit on full data
        for nm, m in base_clfs:
            m.fit(X, y_bin)

        oof_prob = meta_lr.predict_proba(oof_proba)[:, 1]
        oof_pred = (oof_prob >= best_thresh).astype(int)
        eval_mode = 'stacked_oof'

        stage_metrics = {}
        for nm in ['ET', 'RF', 'HGB']:
            prob = base_oof[nm]
            pred = (prob >= 0.5).astype(int)
            stage_metrics[nm] = {
                'acc': {'mean': float(accuracy_score(y_bin, pred))},
                'bacc': {'mean': float(balanced_accuracy_score(y_bin, pred))},
                'f1': {'mean': float(f1_score(y_bin, pred, zero_division=0))},
                'auc': {'mean': float(safe_roc_auc(y_bin, prob))},
                'loss': {'mean': float(log_loss(y_bin, np.clip(prob, 1e-9, 1 - 1e-9), labels=[0, 1]))},
            }

        avg2_prob = np.column_stack([base_oof['ET'], base_oof['RF']]).mean(axis=1)
        avg3_prob = np.column_stack([base_oof['ET'], base_oof['RF'], base_oof['HGB']]).mean(axis=1)
        for nm, prob in [('AVG2', avg2_prob), ('AVG3', avg3_prob), ('STACK', oof_prob)]:
            pred = (prob >= (best_thresh if nm == 'STACK' else 0.5)).astype(int)
            stage_metrics[nm] = {
                'acc': {'mean': float(accuracy_score(y_bin, pred))},
                'bacc': {'mean': float(balanced_accuracy_score(y_bin, pred))},
                'f1': {'mean': float(f1_score(y_bin, pred, zero_division=0))},
                'auc': {'mean': float(safe_roc_auc(y_bin, prob))},
                'loss': {'mean': float(log_loss(y_bin, np.clip(prob, 1e-9, 1 - 1e-9), labels=[0, 1]))},
            }
    else:
        print("  ⚠ Not enough minority samples for stacked CV; using a single-model fallback.")
        single_model = ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        single_model.fit(X, y_bin)
        base_clfs = [('ET', single_model)]
        oof_proba = single_model.predict_proba(X)[:, 1].reshape(-1, 1)
        meta_lr = LogisticRegression(
            penalty='l2', solver='lbfgs', class_weight='balanced',
            max_iter=2000, random_state=42
        )
        meta_lr.fit(oof_proba, y_bin)
        best_thresh = 0.5
        oof_prob = meta_lr.predict_proba(oof_proba)[:, 1]
        oof_pred = (oof_prob >= best_thresh).astype(int)
        eval_mode = 'proxy_fallback'
        stage_metrics = {
            'ET': {
                'acc': {'mean': float(accuracy_score(y_bin, oof_pred))},
                'bacc': {'mean': float(balanced_accuracy_score(y_bin, oof_pred))},
                'f1': {'mean': float(f1_score(y_bin, oof_pred, zero_division=0))},
                'auc': {'mean': float(safe_roc_auc(y_bin, oof_prob))},
                'loss': {'mean': float(log_loss(y_bin, np.clip(oof_prob, 1e-9, 1 - 1e-9), labels=[0, 1]))},
            }
        }

    oof_acc   = accuracy_score(y_bin, oof_pred)
    oof_f1    = f1_score(y_bin, oof_pred, zero_division=0)
    oof_auc   = safe_roc_auc(y_bin, oof_prob)
    oof_mcc   = matthews_corrcoef(y_bin, oof_pred)
    oof_bacc  = balanced_accuracy_score(y_bin, oof_pred)
    oof_cm    = confusion_matrix(y_bin, oof_pred, labels=[0, 1])

    # Regressors
    et_bw  = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1,
                max_features='sqrt', random_state=42, n_jobs=-1)
    rf_bw  = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                max_features='sqrt', random_state=42, n_jobs=-1)
    hgb_bw = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    et_fl  = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1)
    rf_fl  = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
    hgb_fl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    et_fu  = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1)
    rf_fu  = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
    hgb_fu = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)

    REG_W = [0.40, 0.35, 0.25]
    for m in [et_bw,rf_bw,hgb_bw]: m.fit(X_res, y_bw_res)
    for m in [et_fl,rf_fl,hgb_fl]: m.fit(X_res, y_fl_res)
    for m in [et_fu,rf_fu,hgb_fu]: m.fit(X_res, y_fu_res)

    # OOF regressor
    oof_bw = np.zeros(len(X_res))
    for ti,vi in kf.split(X_res):
        fold_p = []
        for m in [et_bw,rf_bw,hgb_bw]:
            mc = deepcopy(m); mc.fit(X_res.iloc[ti], y_bw_res[ti])
            fold_p.append(mc.predict(X_res.iloc[vi]))
        oof_bw[vi] = np.dot(np.column_stack(fold_p), REG_W)
    oof_r2_bw  = r2_score(y_bw_res, oof_bw)
    oof_mae_bw = mean_absolute_error(y_bw_res, oof_bw)

    # Package
    model_bundle = {
        'base_clfs': base_clfs, 'meta_lr': meta_lr, 'best_thresh': best_thresh,
        'et_bw':et_bw,'rf_bw':rf_bw,'hgb_bw':hgb_bw,
        'et_fl':et_fl,'rf_fl':rf_fl,'hgb_fl':hgb_fl,
        'et_fu':et_fu,'rf_fu':rf_fu,'hgb_fu':hgb_fu,
        'REG_W': REG_W, 'X_cols': X.columns.tolist(),
        'n_train': len(df_train), 'n_res': res_mask.sum(),
    }
    metrics = {
        'oof_acc': float(oof_acc), 'oof_f1': float(oof_f1),
        'oof_auc': float(oof_auc), 'oof_mcc': float(oof_mcc),
        'oof_bacc': float(oof_bacc), 'oof_r2_bw': float(oof_r2_bw),
        'oof_mae_bw': float(oof_mae_bw),
        'oof_cm': oof_cm.tolist(),
        'oof_proba': oof_prob.tolist(),
        'y_bin': y_bin.tolist(),
        'best_thresh': float(best_thresh),
        'eval_mode': eval_mode,
        'stage_metrics': stage_metrics,
    }
    return model_bundle, metrics, X, y_bin, oof_proba


# ══════════════════════════════════════════════════════════════════
# UNCERTAINTY SAMPLING — pick next combos to simulate
# ══════════════════════════════════════════════════════════════════
def get_uncertain_combos(model_bundle, df_known, n=20):
    """
    Query strategy: pick combos where the ensemble has MAXIMUM UNCERTAINTY.
    Uncertainty = probability closest to 0.5 (model can't decide).
    Also prioritise combos near regime boundaries (high information gain).

    Returns DataFrame with N most uncertain untested combinations.
    """
    known_keys = set(zip(df_known.LG.round(3), df_known.WR.round(3), df_known.WF.round(3)))

    # Build all untested combos
    untested = []
    for lg in LG_VALS:
        for wr in WR_VALS:
            for wf in WF_VALS:
                key = (round(lg,3), round(wr,3), round(wf,3))
                if key not in known_keys:
                    untested.append({'LG':lg,'WR':wr,'WF':wf})

    if not untested:
        return pd.DataFrame()

    df_u = pd.DataFrame(untested)
    X_u  = build_features(df_u)

    # Get ensemble probabilities
    base_probas = np.column_stack([
        m.predict_proba(X_u)[:,1] for _,m in model_bundle['base_clfs']
    ])
    stack_prob = model_bundle['meta_lr'].predict_proba(base_probas)[:,1]

    # Uncertainty score: entropy of prediction
    # High entropy = model is uncertain → most useful to simulate
    p = np.clip(stack_prob, 1e-9, 1-1e-9)
    entropy = -(p * np.log(p) + (1-p) * np.log(1-p))

    # Also score by BW potential (predicted high BW is valuable)
    res_pred = (stack_prob >= model_bundle['best_thresh']).astype(int)
    bw_pred  = np.zeros(len(df_u))
    if res_pred.sum() > 0:
        X_res_u = X_u[res_pred==1]
        bw_pred[res_pred==1] = np.dot(
            np.column_stack([
                model_bundle['et_bw'].predict(X_res_u),
                model_bundle['rf_bw'].predict(X_res_u),
                model_bundle['hgb_bw'].predict(X_res_u)
            ]), model_bundle['REG_W'])

    # Final score: 70% uncertainty + 30% predicted BW (explore + exploit)
    bw_norm = bw_pred / (bw_pred.max() + 1e-6)
    score   = 0.70 * entropy + 0.30 * bw_norm

    df_u['uncertainty']   = entropy
    df_u['pred_resonant'] = res_pred
    df_u['pred_BW']       = bw_pred.round(4)
    df_u['score']         = score

    # Top N by score
    top = df_u.nlargest(n, 'score').reset_index(drop=True)
    top['rank']   = range(1, len(top)+1)
    top['reason'] = top.apply(lambda r:
        ('HIGH UNCERTAINTY' if r.uncertainty > 0.6 else
         ('REGIME BOUNDARY' if 10.5 < r.WR < 11.5 else
          ('HIGH PRED BW' if r.pred_BW > 1.5 else 'UNKNOWN ZONE'))), axis=1)

    return top


# ══════════════════════════════════════════════════════════════════
# PREDICT FUNCTION (for use after training)
# ══════════════════════════════════════════════════════════════════
def predict(model_bundle, lg, wr, wf, df_known=None, verbose=True):
    lg, wr, wf = float(lg), float(wr), float(wf)

    # Lookup in known data
    if df_known is not None:
        match = df_known[(abs(df_known.LG-lg)<0.001)&
                          (abs(df_known.WR-wr)<0.001)&
                          (abs(df_known.WF-wf)<0.001)]
        if len(match) > 0:
            row = match.iloc[0]
            result = {'resonant':int(row.BW>0),'FL':float(row.FL),'FU':float(row.FU),
                       'BW':float(row.BW),'source':'KNOWN DATA (exact)','confidence':100.0}
            if verbose:
                print(f"  ✅ [KNOWN] L={lg} WR={wr} WF={wf}")
                print(f"     BW={result['BW']:.4f} GHz  FL={result['FL']:.4f}  FU={result['FU']:.4f}")
            return result

    # ML prediction
    feat = build_features(pd.DataFrame({'LG':[lg],'WR':[wr],'WF':[wf]}))
    base_p = np.column_stack([m.predict_proba(feat)[:,1] for _,m in model_bundle['base_clfs']])
    prob   = float(model_bundle['meta_lr'].predict_proba(base_p)[:,1])
    res_   = int(prob >= model_bundle['best_thresh'])
    fl_, fu_, bw_ = 0.0, 0.0, 0.0
    if res_:
        fl_ = float(np.dot([m.predict(feat)[0] for m in [model_bundle['et_fl'],model_bundle['rf_fl'],model_bundle['hgb_fl']]], model_bundle['REG_W']))
        fu_ = float(np.dot([m.predict(feat)[0] for m in [model_bundle['et_fu'],model_bundle['rf_fu'],model_bundle['hgb_fu']]], model_bundle['REG_W']))
        bw_ = float(np.dot([m.predict(feat)[0] for m in [model_bundle['et_bw'],model_bundle['rf_bw'],model_bundle['hgb_bw']]], model_bundle['REG_W']))
        bw_ = max(bw_, 0)
    result = {'resonant':res_,'FL':round(fl_,4),'FU':round(fu_,4),'BW':round(bw_,4),
               'source':f'ML ENSEMBLE (confidence={prob*100:.1f}%)','confidence':prob*100}
    if verbose:
        print(f"  🔮 [ML] L={lg} WR={wr} WF={wf}")
        print(f"     Resonant={bool(res_)}  BW={bw_:.4f} GHz  FL={fl_:.4f}  FU={fu_:.4f}")
        print(f"     Confidence={prob*100:.1f}%  Source={result['source']}")
    return result


# ══════════════════════════════════════════════════════════════════
# GENERATE DIAGNOSTIC FIGURE
# ══════════════════════════════════════════════════════════════════
def make_figure(metrics_all_rounds, df_all, model_bundle, round_idx, oof_proba, y_bin):
    BG="#ffffff"; PANEL="#f7f7f7"; BORDER="#d0d0d0"
    CP={'acc':'#0066cc','f1':'#008a3d','r2':'#b7791f','new':'#c2410c','bw':'#7c3aed'}

    def dark_ax(ax):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values(): s.set_edgecolor(BORDER)
        ax.tick_params(colors='#111111',labelsize=7)
        ax.xaxis.label.set_color('#111111'); ax.yaxis.label.set_color('#111111')
        ax.title.set_color('#111111')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'axes.grid':True,'grid.alpha':0.2,'figure.dpi':110})

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), facecolor=BG)
    m = metrics_all_rounds[-1]
    fig.suptitle(
        f"Active Learning — Round {round_idx} | "
        f"OOF Acc={m['oof_acc']*100:.2f}%  F1={m['oof_f1']:.4f}  "
        f"BW R²={m['oof_r2_bw']:.4f} | n_train={model_bundle['n_train']}",
        fontsize=12, color='black', y=0.99, fontweight='bold')

    # 1. Accuracy across rounds
    ax=axes[0,0]; dark_ax(ax)
    rounds=[m['round'] for m in metrics_all_rounds]
    accs  =[m['oof_acc']*100 for m in metrics_all_rounds]
    f1s   =[m['oof_f1']*100 for m in metrics_all_rounds]
    ntrain=[m['n_train'] for m in metrics_all_rounds]
    ax.plot(rounds, accs, 'o-', color=CP['acc'], lw=2, ms=7, label='OOF Acc %')
    ax.plot(rounds, f1s,  's-', color=CP['f1'],  lw=2, ms=7, label='F1×100')
    ax.axhline(y=98, color=CP['new'], lw=1.5, ls='--', label='98% target')
    for r,a,n in zip(rounds,accs,ntrain):
        ax.annotate(f'{a:.1f}%\n(n={n})', (r,a), textcoords='offset points',
                    xytext=(0,8), ha='center', fontsize=6.5, color='white')
    ax.set_xlabel('Round'); ax.set_ylabel('Score'); ax.set_title('Accuracy Across AL Rounds')
    ax.legend(fontsize=6.5, facecolor=PANEL, labelcolor='black')

    # 2. BW R² across rounds
    ax=axes[0,1]; dark_ax(ax)
    r2s=[m['oof_r2_bw'] for m in metrics_all_rounds]
    ax.plot(rounds, r2s, 'o-', color=CP['r2'], lw=2, ms=7)
    for r,v in zip(rounds,r2s): ax.annotate(f'{v:.4f}', (r,v), textcoords='offset points',
                                              xytext=(0,6), ha='center', fontsize=6.5, color='white')
    ax.set_xlabel('Round'); ax.set_ylabel('OOF R²'); ax.set_title('BW R² Across AL Rounds')

    # 3. Training data size per round
    ax=axes[0,2]; dark_ax(ax)
    new_counts=[m.get('n_new_this_round',0) for m in metrics_all_rounds]
    ax.bar(rounds, ntrain, color=CP['acc'], alpha=0.7, label='Total train')
    ax.bar(rounds, new_counts, color=CP['new'], alpha=0.9, label='New this round')
    ax.set_xlabel('Round'); ax.set_ylabel('n samples'); ax.set_title('Data Growth per Round')
    ax.legend(fontsize=6.5, facecolor=PANEL, labelcolor='black')

    # 4. Confusion matrix
    ax=axes[0,3]; dark_ax(ax)
    cm_ = np.array(m['oof_cm'])
    import seaborn as sns
    sns.heatmap(cm_, ax=ax, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dead','Resonant'], yticklabels=['Dead','Resonant'],
                annot_kws={'size':16,'fontweight':'bold'}, linewidths=1)
    ax.set_title(f'OOF Confusion Matrix (Round {round_idx})')
    [t.set_color('black') for t in ax.get_xticklabels()+ax.get_yticklabels()]

    # 5. OOF probability distribution
    ax=axes[1,0]; dark_ax(ax)
    oof_p = np.array(m.get('oof_proba', []))
    y_b   = np.array(m.get('y_bin', []))
    if len(oof_p)>0:
        ax.hist(oof_p[y_b==0], bins=25, color='#ffcc00', alpha=0.7, density=True, label='Dead')
        ax.hist(oof_p[y_b==1], bins=25, color='#00d4ff', alpha=0.7, density=True, label='Resonant')
        ax.axvline(x=m['best_thresh'], color='white', lw=2, ls='--', label=f"t={m['best_thresh']:.2f}")
    ax.set_xlabel('Predicted Probability'); ax.set_title('OOF Probability Distribution')
    ax.legend(fontsize=6.5, facecolor=PANEL, labelcolor='black')

    # 6. Data source breakdown
    ax=axes[1,1]; dark_ax(ax)
    src_counts = df_all['source'].value_counts()
    colors_src = plt.cm.plasma(np.linspace(0.3,0.9,len(src_counts)))  # type:ignore
    ax.barh(src_counts.index, src_counts.values, color=colors_src, alpha=0.85)
    for i,v in enumerate(src_counts.values):
        ax.text(v+0.5, i, str(v), va='center', color='black', fontsize=8)
    ax.set_xlabel('n rows'); ax.set_title('Data by Source (Original + New Rounds)')

    # 7. Resonance rate per L(Gnd)
    ax=axes[1,2]; dark_ax(ax)
    for src, col in [('original','#00d4ff'), *[(f'round_{i}','#ff4d6d') for i in range(round_idx)]]:
        sub = df_all[df_all.source==src]
        if len(sub)==0: continue
        lg_rates = [sub[abs(sub.LG-lg)<0.01].BW.gt(0).mean()*100
                     if (abs(sub.LG-lg)<0.01).sum()>0 else np.nan
                     for lg in LG_VALS]
        ax.plot(LG_VALS, lg_rates, 'o-', color=col, ms=4, lw=1.5, alpha=0.8, label=src)
    ax.set_xlabel('L(Gnd)'); ax.set_ylabel('Resonance Rate (%)'); ax.set_title('Resonance Rate by L(Gnd)')
    ax.legend(fontsize=6, facecolor=PANEL, labelcolor='black')

    # 8. Coverage map: which (LG, WR) combos have been tested
    ax=axes[1,3]; dark_ax(ax)
    cov = np.zeros((len(LG_VALS), len(WR_VALS)))
    for ri,lg in enumerate(LG_VALS):
        for ci,wr in enumerate(WR_VALS):
            n_orig = df_all[(df_all.source=='original')&(abs(df_all.LG-lg)<0.01)&(abs(df_all.WR-wr)<0.01)].shape[0]
            n_new  = df_all[(df_all.source!='original')&(abs(df_all.LG-lg)<0.01)&(abs(df_all.WR-wr)<0.01)].shape[0]
            cov[ri,ci] = 1 if n_orig>0 else (2 if n_new>0 else 0)
    from matplotlib.colors import ListedColormap
    cmap_ = ListedColormap(['#1a1a2e','#00d4ff','#ff4d6d'])
    ax.imshow(cov, aspect='auto', cmap=cmap_, origin='lower', vmin=0, vmax=2,
              extent=[0,len(WR_VALS),0,len(LG_VALS)])
    ax.set_xticks(np.arange(0,len(WR_VALS),3)+0.5)
    ax.set_xticklabels([f"{WR_VALS[i]:.1f}" for i in range(0,len(WR_VALS),3)], rotation=45, fontsize=5.5)
    ax.set_yticks(np.arange(len(LG_VALS))+0.5)
    ax.set_yticklabels([str(l) for l in LG_VALS], fontsize=7)
    ax.set_title('Coverage Map\nBlue=original  Red=new rounds  Dark=untested')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'al_round_{round_idx}.png')
    plt.savefig(fig_path, facecolor=BG, dpi=110, bbox_inches='tight')
    plt.close()
    return fig_path


# ══════════════════════════════════════════════════════════════════
# MAIN ACTIVE LEARNING LOOP
# ══════════════════════════════════════════════════════════════════
def run_round(round_idx, new_data_csv=None):
    """
    Execute one round of active learning.

    round_idx    : 0 = first run, 1+ = after you've collected new sim data
    new_data_csv : path to CSV with new simulation results (for round>0)
    """
    print(f"\n{'='*70}")
    print(f"  ACTIVE LEARNING — ROUND {round_idx}")
    print(f"{'='*70}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    history = load_history()

    # ── Ingest new simulation results (if any) ───────────────────
    if round_idx > 0 and new_data_csv:
        print(f"\n  Loading new simulation results from: {new_data_csv}")
        df_new = pd.read_csv(new_data_csv)

        # Validate format
        required = ['LG','WR','WF','FL','FU','BW']
        missing  = [c for c in required if c not in df_new.columns]
        if missing:
            print(f"  ❌ ERROR: CSV missing columns: {missing}")
            print(f"     Required: {required}")
            print(f"     Your CSV has: {df_new.columns.tolist()}")
            return

        # Store in history
        df_new['source'] = f'round_{round_idx-1}'
        if 'all_new_data' not in history: history['all_new_data'] = []
        while len(history['all_new_data']) < round_idx:
            history['all_new_data'].append([])
        history['all_new_data'][round_idx-1] = df_new.to_dict('records')
        print(f"  ✅ Ingested {len(df_new)} new simulation results")
        save_history(history)

    # ── Load all data ─────────────────────────────────────────────
    df_all = load_all_data(history)
    n_orig = (df_all.source == 'original').sum()
    n_new  = len(df_all) - n_orig

    print(f"\n  DATA SUMMARY:")
    print(f"    Original simulation rows : {n_orig}")
    print(f"    New rows (all rounds)    : {n_new}")
    print(f"    Total                    : {len(df_all)}")
    print(f"    Resonant                 : {(df_all.BW>0).sum()} ({(df_all.BW>0).mean()*100:.1f}%)")

    # ── 80/20 split for validation ────────────────────────────────
    print(f"\n  TRAIN/VALIDATION SPLIT (80/20, minority-safe):")
    df_train, df_val, split_info = make_safe_test_split(df_all, test_size=TEST_SIZE, seed=42)
    print(f"    Train: {len(df_train)} rows  Resonant={split_info['train_res']}  Dead={split_info['train_dead']}")
    print(f"    Val  : {len(df_val)} rows   Resonant={split_info['test_res']}  Dead={split_info['test_dead']}")
    if split_info['test_dead'] == 0:
        print("    ⚠ Warning: test set still has no dead samples; metrics on the minority class remain limited.")

    # ── Epoch-based test monitor for requested plots ───────────────
    print(f"\n  TRAINING 20-EPOCH MONITOR on 80% ({len(df_train)} rows)...")
    monitor = train_epoch_monitor(df_train, df_val, round_idx)
    print(f"    20% Test Accuracy : {monitor['test_acc']*100:.2f}%")
    print(f"    20% Test Confusion Matrix:\n    {np.array(monitor['test_cm'])}")
    print(f"\n{monitor['test_report']}")
    print(f"    Loss plot saved   : {monitor['fig_paths']['loss']}")
    print(f"    Accuracy saved    : {monitor['fig_paths']['accuracy']}")
    print(f"    Confusion saved   : {monitor['fig_paths']['confusion']}")
    print(f"    Epoch history fig : {monitor['epoch_history_fig']}")
    print(f"    Epoch history CSV : {monitor['epoch_csv']}")

    gradual = train_gradual_learning_curve(df_train, df_val, round_idx)
    print(f"    Gradual curve fig : {gradual['fig_path']}")
    print(f"    Gradual curve CSV : {gradual['csv_path']}")

    # ── Train active-learning ensemble on 90% ─────────────────────
    print(f"\n  TRAINING on 80% ({len(df_train)} rows)...")
    model_bundle, oof_metrics, X_train, y_train, oof_proba_arr = train_model(df_train)

    # ── Validate on 10% ───────────────────────────────────────────
    print(f"\n  VALIDATING on 20% ({len(df_val)} rows)...")
    X_val  = build_features(df_val)
    y_val  = (df_val.BW > 0).astype(int).values
    y_bw_val = df_val.BW.values

    base_proba_val = np.column_stack([
        m.predict_proba(X_val)[:,1] for _,m in model_bundle['base_clfs']
    ])
    val_prob  = model_bundle['meta_lr'].predict_proba(base_proba_val)[:,1]
    val_pred  = (val_prob >= model_bundle['best_thresh']).astype(int)
    val_acc   = accuracy_score(y_val, val_pred)
    val_f1    = f1_score(y_val, val_pred, zero_division=0)
    val_auc   = safe_roc_auc(y_val, val_prob)
    val_bacc  = balanced_accuracy_score(y_val, val_pred)
    val_cm    = confusion_matrix(y_val, val_pred)

    # BW on resonant val rows
    res_val = (y_val == 1)
    if res_val.sum() > 0:
        X_res_val = X_val[res_val]
        bw_pred_val = np.dot(np.column_stack([
            model_bundle['et_bw'].predict(X_res_val),
            model_bundle['rf_bw'].predict(X_res_val),
            model_bundle['hgb_bw'].predict(X_res_val)
        ]), model_bundle['REG_W'])
        val_r2_bw  = r2_score(y_bw_val[res_val], bw_pred_val)
        val_mae_bw = mean_absolute_error(y_bw_val[res_val], bw_pred_val)
        val_within01 = (np.abs(bw_pred_val - y_bw_val[res_val]) < 0.1).mean() * 100
    else:
        val_r2_bw=val_mae_bw=val_within01=0.0

    print(f"\n  {'─'*55}")
    print(f"  RESULTS — ROUND {round_idx}")
    print(f"  {'─'*55}")
    train_eval_label = "TRAIN (OOF CV, no leakage)" if oof_metrics.get('eval_mode') == 'stacked_oof' else "TRAIN (proxy fit; stacked CV unavailable)"
    print(f"  {train_eval_label}:")
    print(f"    Accuracy     : {oof_metrics['oof_acc']*100:.2f}%")
    print(f"    F1           : {oof_metrics['oof_f1']:.4f}")
    print(f"    AUC-ROC      : {oof_metrics['oof_auc']:.4f}")
    print(f"    Balanced Acc : {oof_metrics['oof_bacc']*100:.2f}%")
    print(f"    BW R² (OOF)  : {oof_metrics['oof_r2_bw']:.4f}")
    print(f"    BW MAE (OOF) : {oof_metrics['oof_mae_bw']:.4f} GHz")
    if oof_metrics.get('eval_mode') != 'stacked_oof':
        print("    ⚠ Note: these are proxy fit metrics, not honest out-of-fold scores.")
    print(f"\n  VALIDATION (held-out 20%):")
    print(f"    Accuracy     : {val_acc*100:.2f}%")
    print(f"    F1           : {val_f1:.4f}")
    print(f"    AUC-ROC      : {val_auc:.4f}")
    print(f"    Balanced Acc : {val_bacc*100:.2f}%")
    print(f"    BW R²        : {val_r2_bw:.4f}")
    print(f"    BW MAE       : {val_mae_bw:.4f} GHz")
    print(f"    BW within 0.1: {val_within01:.1f}%")
    print(f"\n  CONFUSION MATRIX (Validation):")
    print(f"    {val_cm}")
    print(f"\n{safe_classification_report(y_val, val_pred)}")

    # ── Repeated resampling evaluation ───────────────────────────
    print(f"\n  REPEATED-SPLIT EVALUATION (3 resamples, 80/20 each):")
    per_run, repeat_summary = repeated_split_evaluation(df_all, round_idx=round_idx, n_repeats=3, test_size=TEST_SIZE, seed=42)
    print(f"    Averaged test accuracy     : {repeat_summary['val_acc']['mean']*100:.2f}% ± {repeat_summary['val_acc']['std']*100:.2f}")
    print(f"    Averaged balanced accuracy : {repeat_summary['val_bacc']['mean']*100:.2f}% ± {repeat_summary['val_bacc']['std']*100:.2f}")
    print(f"    Averaged F1                : {repeat_summary['val_f1']['mean']:.4f} ± {repeat_summary['val_f1']['std']:.4f}")
    print(f"    Averaged AUC-ROC           : {repeat_summary['val_auc']['mean']:.4f} ± {repeat_summary['val_auc']['std']:.4f}")
    print(f"    Averaged BW R²             : {repeat_summary['val_r2_bw']['mean']:.4f} ± {repeat_summary['val_r2_bw']['std']:.4f}")
    print(f"    Averaged BW MAE            : {repeat_summary['val_mae_bw']['mean']:.4f} ± {repeat_summary['val_mae_bw']['std']:.4f}")
    print(f"    Summary plot saved         : {repeat_summary['fig_path']}")

    overfitting_flag = (oof_metrics['oof_acc'] - val_acc) > 0.05
    if overfitting_flag:
        print(f"  ⚠ OVERFITTING DETECTED: OOF acc={oof_metrics['oof_acc']*100:.1f}% vs Val={val_acc*100:.1f}%")
        print(f"    → Need more training data OR reduce model complexity")
    else:
        print(f"  ✅ No significant overfitting (gap={( oof_metrics['oof_acc']-val_acc)*100:.1f}%)")

    # ── Save metrics to history ───────────────────────────────────
    round_metrics = {
        'round': round_idx,
        'timestamp': datetime.now().isoformat(),
        'n_train': len(df_train),
        'n_val': len(df_val),
        'n_new_this_round': len(df_new) if (round_idx>0 and new_data_csv) else 0, # type:ignore
        'test_acc_10pct': float(monitor['test_acc']),
        'test_cm_10pct': monitor['test_cm'],
        'epoch_loss': monitor['train_losses'],
        'epoch_train_acc': monitor['train_accs'],
        'epoch_test_acc': monitor['test_accs'],
        'epoch_monitor_figs': monitor['fig_paths'],
        'repeated_eval': per_run,
        'repeated_eval_summary': repeat_summary,
        'eval_mode': oof_metrics.get('eval_mode', 'stacked_oof'),
        'oof_acc': oof_metrics['oof_acc'],
        'oof_f1': oof_metrics['oof_f1'],
        'oof_auc': oof_metrics['oof_auc'],
        'oof_bacc': oof_metrics['oof_bacc'],
        'oof_r2_bw': oof_metrics['oof_r2_bw'],
        'oof_mae_bw': oof_metrics['oof_mae_bw'],
        'oof_cm': oof_metrics['oof_cm'],
        'oof_proba': oof_metrics['oof_proba'],
        'y_bin': oof_metrics['y_bin'],
        'best_thresh': oof_metrics['best_thresh'],
        'val_acc': float(val_acc),
        'val_f1': float(val_f1),
        'val_r2_bw': float(val_r2_bw),
        'val_mae_bw': float(val_mae_bw),
    }
    history['metrics_per_round'].append(round_metrics)
    save_history(history)

    # ── Retrain on FULL data for best model ──────────────────────
    print(f"\n  Retraining on FULL {len(df_all)} rows for deployment...")
    model_final, final_metrics, _, _, _ = train_model(df_all)
    pkl_path = os.path.join(OUTPUT_DIR, f'antenna_model_round_{round_idx}.pkl')
    with open(pkl_path,'wb') as f: pickle.dump({'model':model_final,'df_all':df_all,'round':round_idx}, f)
    print(f"  ✅ Model saved: {pkl_path}")

    if 'stage_metrics' in final_metrics:
        cmp_fig = make_technique_comparison_figure(round_idx, final_metrics['stage_metrics'])
        acc_loss_figs = make_technique_accuracy_loss_figures(round_idx, final_metrics['stage_metrics'])
        print(f"  Technique comparison saved: {cmp_fig}")
        print(f"  Technique accuracy saved  : {acc_loss_figs['accuracy']}")
        print(f"  Technique loss saved      : {acc_loss_figs['loss']}")
        print(f"\n  TECHNIQUE LADDER (full-data OOF):")
        ladder_order = ['ET', 'RF', 'HGB', 'AVG2', 'AVG3', 'STACK']
        ladder_label = {
            'ET': 'ExtraTrees',
            'RF': 'RandomForest',
            'HGB': 'HistGB',
            'AVG2': 'ET+RF',
            'AVG3': 'ET+RF+HGB',
            'STACK': 'Stacked',
        }
        for stage in ladder_order:
            if stage in final_metrics['stage_metrics']:
                st = final_metrics['stage_metrics'][stage]
                print(f"    {ladder_label[stage]:<12} Acc={st['acc']['mean']*100:.2f}%  "
                      f"BAcc={st['bacc']['mean']*100:.2f}%  F1={st['f1']['mean']:.4f}  AUC={st['auc']['mean']:.4f}")

    # ── Suggest next combos to simulate ──────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  NEXT SIMULATIONS TO RUN (Round {round_idx+1})")
    print(f"  {'─'*55}")
    print(f"  These {N_SUGGEST} combos give maximum information gain:")
    print(f"  (70% uncertain + 30% predicted high-BW)\n")

    suggestions = get_uncertain_combos(model_final, df_all, n=N_SUGGEST)

    if len(suggestions) > 0:
        print(f"  {'Rank':>4}  {'L(Gnd)':>7} {'W(Rect)':>8} {'W(Feed)':>8}  "
              f"{'Pred_BW':>9}  {'Uncertain':>9}  Reason")
        print(f"  {'─'*70}")
        for _, row in suggestions.iterrows():
            print(f"  {int(row['rank']):>4}  {row.LG:>7.1f} {row.WR:>8.2f} {row.WF:>8.1f}  "
                  f"{row.pred_BW:>9.4f}  {row.uncertainty:>9.4f}  {row.reason}")

        sug_path = os.path.join(OUTPUT_DIR, f'suggested_round_{round_idx}.csv')
        suggestions[['rank','LG','WR','WF','pred_BW','uncertainty','reason']].to_csv(sug_path, index=False)
        print(f"\n  ✅ Suggestions saved: {sug_path}")
        print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │  NEXT STEP:                                                 │")
        print(f"  │  1. Open: {sug_path:<48} │")
        print(f"  │  2. Run each row in your simulation software                │")
        print(f"  │  3. Fill results into: new_data_round_{round_idx}.csv           │")
        print(f"  │  4. CSV must have columns: LG,WR,WF,FL,FU,BW               │")
        print(f"  │  5. Then run: python active_learning.py --round {round_idx+1:<2}         │")
        print(f"  │             --new_data new_data_round_{round_idx}.csv           │")
        print(f"  └─────────────────────────────────────────────────────────────┘")

        # Also create empty template CSV for user to fill
        template_cols = ['LG','WR','WF','FL','FU','BW']
        template = suggestions[['LG','WR','WF']].copy()
        template['FL'] = ''
        template['FU'] = ''
        template['BW'] = ''
        tmpl_path = os.path.join(OUTPUT_DIR, f'new_data_round_{round_idx}.csv')
        template.to_csv(tmpl_path, index=False)
        print(f"\n  📋 Empty template created: {tmpl_path}")
        print(f"     Fill in FL, FU, BW columns from your simulation results")

    # ── Generate figure ───────────────────────────────────────────
    fig_path = make_figure(
        history['metrics_per_round'],
        df_all, model_final, round_idx,
        np.array(oof_metrics['oof_proba']),
        np.array(oof_metrics['y_bin']))
    print(f"\n  ✅ Figure saved: {fig_path}")

    return model_final, df_all


# ══════════════════════════════════════════════════════════════════
# RANDOM CHECKER
# ══════════════════════════════════════════════════════════════════
def random_checker(model_bundle, df_all, n=10, seed=42):
    import random as rnd; rnd.seed(seed)
    sample = df_all.sample(n=min(n, len(df_all)), random_state=seed).reset_index(drop=True)
    print(f"\n{'='*65}")
    print(f"  RANDOM CHECKER — {len(sample)} random samples")
    print(f"{'='*65}")
    print(f"  {'#':>3}  {'L':>5} {'WR':>7} {'WF':>6}  {'Act_BW':>8} {'Pred_BW':>9} {'Err':>6}  Status  Source")
    print(f"  {'─'*68}")
    errors=[]
    for i,(_,row) in enumerate(sample.iterrows(),1):
        res = predict(model_bundle, row.LG, row.WR, row.WF, df_all, verbose=False)
        err = abs(res['BW']-row.BW); errors.append(err)
        src = 'KNOWN' if 'KNOWN' in res['source'] else 'ML'
        st  = "✅ EXACT" if err<0.001 else ("✓ OK" if err<0.1 else "⚠ OFF")
        print(f"  {i:>3}  {row.LG:>5.1f} {row.WR:>7.2f} {row.WF:>6.1f}  "
              f"{row.BW:>8.4f} {res['BW']:>9.4f} {err:>6.4f}  {st:>8}  [{src}]")
    print(f"\n  Mean err={np.mean(errors):.6f}  Max err={np.max(errors):.6f}")
    print(f"  Exact(|Δ|<0.001): {sum(e<0.001 for e in errors)}/{len(errors)} = {sum(e<0.001 for e in errors)/len(errors)*100:.0f}%")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Active Learning pipeline for antenna bandwidth prediction')
    parser.add_argument('--round', type=int, default=0,
                        help='Round number (0=first run, 1+=after simulation)')
    parser.add_argument('--new_data', type=str, default=None,
                        help='CSV with new simulation results (for round > 0)')
    parser.add_argument('--check', action='store_true',
                        help='Run random checker after training')
    parser.add_argument('--predict', nargs=3, type=float,
                        metavar=('LG','WR','WF'),
                        help='Predict for a specific (LG, WR, WF)')
    args = parser.parse_args()

    model, df_data = run_round(args.round, args.new_data) # type:ignore

    if args.check:
        random_checker(model, df_data, n=15, seed=99)

    if args.predict:
        lg, wr, wf = args.predict
        print(f"\n{'='*55}")
        print(f"  PREDICTION REQUEST: L={lg} WR={wr} WF={wf}")
        print(f"{'='*55}")
        predict(model, lg, wr, wf, df_data, verbose=True)

    print(f"\n{'='*70}")
    print(f"  WORKFLOW COMPLETE — ROUND {args.round}")
    print(f"{'='*70}")
