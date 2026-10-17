import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

def plot_confusion_matrix():
    preds_path = Path("results/preds/cba_phenonet_42.npz")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
        
    data = np.load(preds_path)
    y_true = data["y_true"].flatten()
    y_pred = data["y_pred"].flatten()
    
    classes = ["Fallow", "Vegetative", "Reproductive", "Senescence"]
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    
    # Custom color map for the heatmap
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 11, "weight": "bold"},
                ax=ax, vmin=0, vmax=100)
                
    # Modify the text to include the '%' sign and the absolute count underneath?
    # Or just keep it as percentage to avoid clutter.
    # Let's customize the annotations to show "XX.X%" and "\n(N)"
    for i in range(4):
        for j in range(4):
            text = ax.texts[i*4 + j]
            val = cm_percent[i, j]
            count = cm[i, j]
            text.set_text(f"{val:.1f}%\n({count})")
    
    ax.set_xlabel("Predicted Stage", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Stage", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Temporal Segmentation Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    
    plt.xticks(rotation=45, ha='right', fontsize=10.5)
    plt.yticks(rotation=0, fontsize=10.5)
    
    plt.tight_layout()
    
    out1 = Path("docs/figures/confusion_matrix.png")
    out2 = Path("results/plots/confusion_matrix.png")
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out1}")
    print(f"Saved {out2}")
    
if __name__ == "__main__":
    plot_confusion_matrix()
