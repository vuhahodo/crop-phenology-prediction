"""Generates publication-quality architecture diagram for CBA-PhenoNet.
Updated per Codex review: Increased typography by 25-30%, concise keywords, high legibility.
Saves figure to results/plots/cba_phenonet_architecture.png and docs/figures/cba_phenonet_architecture.png.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14.5, 7.8), dpi=300)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    # Helper function for drawing rounded boxes with large, legible typography
    def draw_box(x, y, w, h, title, subtitle="", color="#EBF3FB", border="#2B6CB0", title_color="#1A365D", title_fs=12.5, sub_fs=10.5):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                             facecolor=color, edgecolor=border, linewidth=1.6, zorder=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=title_color, zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=sub_fs, fontweight="medium", color="#4A5568", zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=title_fs, fontweight="bold", color=title_color, zorder=3)

    def draw_arrow(x1, y1, x2, y2, color="#4A5568", text="", text_fs=9.5):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                color=color, linewidth=2.0, zorder=4)
        ax.add_patch(arrow)
        if text:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.16, text, ha="center", va="bottom", fontsize=text_fs, fontweight="bold", color="#2D3748", zorder=5)

    # 1. INPUT LAYER
    input_group = FancyBboxPatch((0.4, 1.0), 3.1, 5.8, boxstyle="round,pad=0.1",
                                 facecolor="#F7FAFC", edgecolor="#CBD5E0", linestyle="--", linewidth=1.4, zorder=1)
    ax.add_patch(input_group)
    ax.text(1.95, 6.55, "Input Representation", ha="center", fontsize=13, fontweight="bold", color="#2D3748")

    draw_box(0.6, 4.4, 2.7, 1.4, "Sentinel-2 NDVI", "Smoothed S-G series (T, 1)", color="#E6FFFA", border="#319795", title_color="#234E52")
    draw_box(0.6, 1.6, 2.7, 1.4, "Cyclical Day-of-Year", "sin & cos harmonics (T, 2)", color="#FEFCBF", border="#D69E2E", title_color="#744210")

    # 2. EMBEDDING & POSITIONAL ENCODING
    draw_box(4.1, 4.4, 2.3, 1.3, "Linear Projection", "NDVI -> d_model (64)", color="#EDF2F7", border="#718096")
    draw_box(4.1, 1.6, 2.3, 1.3, "DOY Pos-Encoding", "Date-Aware Linear Proj", color="#FEFCBF", border="#D69E2E", title_color="#744210")

    draw_arrow(3.3, 5.05, 4.1, 5.05)
    draw_arrow(3.3, 2.25, 4.1, 2.25)

    # Sum circle (+)
    sum_circle = plt.Circle((5.25, 3.35), 0.32, facecolor="#FFFFFF", edgecolor="#2B6CB0", linewidth=1.8, zorder=3)
    ax.add_patch(sum_circle)
    ax.text(5.25, 3.35, "+", ha="center", va="center", fontsize=18, fontweight="bold", color="#2B6CB0", zorder=4)

    draw_arrow(5.25, 4.4, 5.25, 3.7)
    draw_arrow(5.25, 2.9, 5.25, 3.02)

    # 3. CBA-PhenoNet ENCODER LAYERS (Container)
    encoder_group = FancyBboxPatch((7.0, 0.8), 3.6, 6.2, boxstyle="round,pad=0.1",
                                   facecolor="#F0F4F8", edgecolor="#2B6CB0", linewidth=2.2, zorder=1)
    ax.add_patch(encoder_group)
    ax.text(8.8, 6.65, "CBA-PhenoNet Encoder (x2)", ha="center", fontsize=13, fontweight="bold", color="#1A365D")

    draw_arrow(5.6, 3.35, 7.3, 4.9, text="Token Sequence")

    draw_box(7.3, 4.4, 3.0, 1.5, "Self-Attention Block", "Multi-Head Attention (h=4)\nFFN + LayerNorm", color="#FFFFFF", border="#4299E1", title_color="#2B6CB0")
    
    # Phenology Gate inside layer
    draw_box(7.3, 1.4, 3.0, 1.6, "Calendar Gating Prior", "MLP: 2 -> 16 -> 1 -> Sigmoid\ng(t) in (0, 1) per timestep", color="#FFF5F5", border="#E53E3E", title_color="#9B2C2C")

    # Connect DOY directly to Phenology Gate
    ax.annotate("", xy=(7.3, 2.2), xytext=(5.25, 1.6),
                arrowprops=dict(arrowstyle="-|>", color="#DD6B20", lw=2.0, connectionstyle="arc3,rad=-0.18"), zorder=4)
    ax.text(6.25, 1.65, "DOY signal", fontsize=9.5, fontweight="bold", color="#C05621")

    # Gate Multiplication circle (x)
    mult_circle = plt.Circle((8.8, 3.65), 0.30, facecolor="#FFFFFF", edgecolor="#E53E3E", linewidth=1.8, zorder=3)
    ax.add_patch(mult_circle)
    ax.text(8.8, 3.65, "×", ha="center", va="center", fontsize=18, fontweight="bold", color="#E53E3E", zorder=4)

    draw_arrow(8.8, 4.4, 8.8, 4.0)
    draw_arrow(8.8, 3.0, 8.8, 3.32) # from gate up to mult

    # Residual / Output from encoder
    draw_arrow(9.15, 3.65, 11.1, 4.9)

    # 4. CLASSIFICATION & DERIVATION
    output_group = FancyBboxPatch((11.0, 0.8), 3.1, 6.2, boxstyle="round,pad=0.1",
                                  facecolor="#F7FAFC", edgecolor="#CBD5E0", linestyle="--", linewidth=1.4, zorder=1)
    ax.add_patch(output_group)
    ax.text(12.55, 6.65, "Output & Milestone Decoding", ha="center", fontsize=13, fontweight="bold", color="#2D3748")

    draw_box(11.2, 4.3, 2.7, 1.5, "Segmentation Head", "Linear Proj -> 4 Classes\n(Fallow, Veg, Rep, Sen)", color="#EDF2F7", border="#4A5568")

    draw_arrow(12.55, 4.3, 12.55, 3.2)

    draw_box(11.2, 1.4, 2.7, 1.7, "Protocol A Decoding", "Transition 0->1 : SOS\nNDVI Peak : POS\nTransition 3->0 : EOS", color="#E6FFFA", border="#319795", title_color="#234E52")

    # Save to both target locations
    docs_out = Path("docs/figures/cba_phenonet_architecture.png")
    plots_out = Path("results/plots/cba_phenonet_architecture.png")
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    plots_out.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(docs_out, dpi=300, bbox_inches="tight")
    plt.savefig(plots_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated high-legibility architecture diagram at: {docs_out} and {plots_out}")

if __name__ == "__main__":
    create_architecture_diagram()
