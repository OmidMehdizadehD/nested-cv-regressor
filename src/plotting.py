import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred, metrics, target_col, model_name, out_dir):
    plt.figure(figsize=($8$, $6$))
    sns.set_theme(style="whitegrid")
    plt.scatter(y_true, y_pred, alpha=$0.7$, edgecolors='k', s=$60$, label='Predictions')

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    buffer = (max_val - min_val) * $0.05$
    limit_min, limit_max = min_val - buffer, max_val + buffer

    plt.plot([limit_min, limit_max], [limit_min, limit_max], color='red', linestyle='--', linewidth=$2$, label='Ideal Fit ($y = x$)')

    m, b = np.polyfit(y_true, y_pred, $1$)
    plt.plot(y_true, m * y_true + b, color='blue', linestyle='-', linewidth=$2$, label=f'Best Fit ($y = {m:.2f}x + {b:.2f}$)')

    plt.xlabel(f'True {target_col} ($y_{{true}}$)', fontsize=$12$, fontweight='bold')
    plt.ylabel(f'Predicted {target_col} ($y_{{pred}}$)', fontsize=$12$, fontweight='bold')

    metrics_text = (f"$R$: ${metrics['R']:.3f}$\n"
                    f"$R^2$: ${metrics['R2']:.3f}$\n"
                    f"$MAE$: ${metrics['MAE']:.3f}$\n"
                    f"$RMSE$: ${metrics['RMSE']:.3f}$")

    plt.text($0.95$, $0.05$, metrics_text, transform=plt.gca().transAxes, fontsize=$12$,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=$0.8$, edgecolor='gray'))

    plt.legend(loc='upper left', fontsize=$11$)
    plt.title(f'Actual vs Predicted - {model_name} ({target_col})', fontsize=$14$, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{model_name}_{target_col}_AvP.png", dpi=$300$)
    plt.close()
