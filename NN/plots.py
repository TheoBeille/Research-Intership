

import os
import re
import matplotlib.pyplot as plt


def _safe_filename(title: str) -> str:
    name = title.replace(' ', '_')
    name = re.sub(r'[^A-Za-z0-9_.-]', '', name)
    return name


def _ensure_plots_dir(dir_name: str = 'plots') -> str:
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def plot_convergence(res_zero, res_learned,res_learned_primal,label3, title="Convergence",label1="Zero (baseline)",label2="Learned"):
    plt.figure(figsize=(6,4))

    plt.semilogy(res_zero, label=label1, linewidth=2)
    plt.semilogy(res_learned, label=label2, linewidth=2, linestyle='--')
    plt.semilogy(res_learned_primal, label=label3, linewidth=2, linestyle='dotted')
    plt.xlabel("Iteration")
    plt.ylabel("Residual (log scale)")
    plt.title(title)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    
def plot_convergence_2(res_zero,res_learned_primal,label3="learned_primal", title="Convergence",label1="Zero (baseline)"):
    plt.figure(figsize=(6,4))

    plt.loglog(res_zero, label=label1, linewidth=2)
    plt.loglog(res_learned_primal, label=label3, linewidth=2, linestyle='dotted')
    plt.xlabel("Iteration")
    plt.ylabel("Residual (log scale)")
    plt.title(title)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    
    plt.show()
    plt.close()

def train_plot(train_loss_hist,val_loss_hist,title="Training_Validation Errors"):
    plt.figure(figsize=(10, 5))
    plt.semilogy(train_loss_hist, label="Training error",   color="#3B5BA5", linewidth=1.5)
    plt.semilogy(val_loss_hist,   label="Validation error", color="#C07820", linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_dir = _ensure_plots_dir()
    fname = f"{_safe_filename(title)}.pdf"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()

