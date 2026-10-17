import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    print("Exporting tables and figures for paper...")
    
    os.makedirs("paper/figs", exist_ok=True)
    os.makedirs("paper/tables", exist_ok=True)
    
    # 1. Convert CSV Tables to LaTeX
    try:
        df_ablation = pd.read_csv("results/tables/table_ablation.csv")
        with open("paper/tables/table_ablation.tex", "w") as f:
            f.write(df_ablation.to_latex(index=False, caption="Ablation Study Results", label="tab:ablation", float_format="%.4f"))
        print("  Exported paper/tables/table_ablation.tex")
    except Exception as e:
        print(f"  Skipping table_ablation.csv: {e}")
        
    try:
        df_pvalues = pd.read_csv("results/tables/table_pvalues_full.csv")
        with open("paper/tables/table_pvalues_full.tex", "w") as f:
            f.write(df_pvalues.to_latex(index=False, caption="Statistical Significance and TOST Equivalence", label="tab:pvalues"))
        print("  Exported paper/tables/table_pvalues_full.tex")
    except Exception as e:
        print(f"  Skipping table_pvalues_full.csv: {e}")

    # 2. Re-plot curves as Vector PDF if needed. 
    # Since eval_phase_a.py and b0c_plot_gate.py generated .png files, 
    # the proper way is to modify them to generate .pdf, or we can just 
    # use Pillow to convert (but that loses vectorization). 
    # Best practice is to have the scripts save as .pdf in paper/figs.
    
    print("\nDone. Please ensure that matplotlib scripts (eval_phase_a, b0c_plot_gate) are updated to save to paper/figs as .pdf")

if __name__ == "__main__":
    main()
