import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def agg_rankings():
    data = {
        "Method": ["Permutation Importance", "SHAP"],
        "3": [4.22, 1.67],
        "6": [6.22, 6.44],
        "7": [3.44, 4.89],
        "11": [7.11, 6.11],
        "12": [3.22, 1.78],
        "13": [2.44, 4.33],
        "14": [5.11, 5.89],
    }

    df = pd.DataFrame(data)
    df.set_index("Method", inplace=True)

    mean_ranks = df.mean().sort_values()
    df_melted = df.melt(var_name="Feature", value_name="Rank")

    order = mean_ranks.index

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=150)

    bar = sns.barplot(
        data=df_melted,
        x="Rank",
        y="Feature",
        order=order,
        palette="viridis",
        alpha=0.6,
        errorbar=None,
    )

    sns.stripplot(
        data=df_melted,
        x="Rank",
        y="Feature",
        order=order,
        color="black",
        alpha=0.6,
        size=5,
        jitter=False,
    )

    plt.title("Mean Avg. Ranking: Average across 9 Models", fontsize=16, pad=20)
    plt.xlabel("Avg. Rank (Lower is More Important)", fontsize=12)
    plt.ylabel("Feature ID", fontsize=12)

    for i, feature in enumerate(order):
        avg_val = mean_ranks[feature]
        plt.text(
            avg_val + 0.2,
            i - 0.15,
            f"{avg_val:.2f}",
            va="center",
            color="black",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("./avg_rankings_agg.png")


def shap_rankings():
    data = {
        "Model": [
            "LR",
            "MLP",
            "LSVC",
            "RBF-SVC",
            "Poly-SVC",
            "Sig-SVC",
            "DT",
            "RF",
            "HGBC",
        ],
        "3": [1, 3, 1, 1, 1, 1, 2, 4, 1],
        "6": [7, 5, 6, 7, 9, 6, 7, 6, 5],
        "7": [4, 4, 5, 2, 15, 3, 4, 3, 4],
        "11": [5, 6, 4, 5, 13, 5, 5, 5, 7],
        "12": [2, 1, 2, 3, 2, 2, 1, 1, 2],
        "13": [6, 2, 7, 4, 3, 9, 3, 2, 3],
        "14": [3, 8, 3, 6, 8, 4, 6, 9, 6],
    }

    df = pd.DataFrame(data)
    df.set_index("Model", inplace=True)

    mean_ranks = df.mean().sort_values()
    df_melted = df.melt(var_name="Feature", value_name="Rank")

    order = mean_ranks.index

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=150)

    bar = sns.barplot(
        data=df_melted,
        x="Rank",
        y="Feature",
        order=order,
        palette="viridis",
        alpha=0.6,
        errorbar=None,
    )

    sns.stripplot(
        data=df_melted,
        x="Rank",
        y="Feature",
        order=order,
        color="black",
        alpha=0.6,
        size=5,
        jitter=False,
    )

    plt.title(
        "Mean Absolute SHAP Value Ranking: Average across 9 Models", fontsize=16, pad=20
    )
    plt.xlabel("Rank (Lower is More Important)", fontsize=12)
    plt.ylabel("Feature ID", fontsize=12)

    for i, feature in enumerate(order):
        avg_val = mean_ranks[feature]
        plt.text(
            avg_val + 0.2,
            i - 0.15,
            f"{avg_val:.2f}",
            va="center",
            color="black",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("./avg_rankings_shap.png")


def importance_rankings():
    data = {
        "Model": [
            "LR",
            "MLP",
            "LSVC",
            "RBF-SVC",
            "Poly-SVC",
            "Sig-SVC",
            "DT",
            "RF",
            "HGBC",
        ],
        "3": [6, 5, 6, 4, 6, 6, 2, 2, 1],
        "6": [7, 4, 7, 7, 8, 7, 6, 6, 4],
        "7": [2, 3, 2, 1, 11, 2, 4, 3, 3],
        "11": [5, 8, 5, 5, 12, 3, 7, 9, 10],
        "12": [1, 2, 1, 3, 10, 1, 1, 4, 6],
        "13": [3, 1, 3, 2, 2, 5, 3, 1, 2],
        "14": [4, 7, 4, 6, 5, 4, 5, 6, 5],
    }

    df = pd.DataFrame(data)
    df.set_index("Model", inplace=True)

    mean_ranks = df.mean().sort_values()
    df_melted = df.melt(var_name="Feature", value_name="Rank")

    order = mean_ranks.index

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6), dpi=150)

    bar = sns.barplot(
        data=df_melted,
        x="Rank",
        y="Feature",
        order=order,
        palette="viridis",
        alpha=0.6,
        errorbar=None,
    )

    sns.stripplot(
        data=df_melted,
        x="Rank",
        y="Feature",
        order=order,
        color="black",
        alpha=0.6,
        size=5,
        jitter=False,
    )

    plt.title(
        "Permutation Importance: Average Ranking across 9 Models", fontsize=16, pad=20
    )
    plt.xlabel("Rank (Lower is More Important)", fontsize=12)
    plt.ylabel("Feature ID", fontsize=12)

    for i, feature in enumerate(order):
        avg_val = mean_ranks[feature]
        plt.text(
            avg_val + 0.2,
            i - 0.15,
            f"{avg_val:.2f}",
            va="center",
            color="black",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("./avg_rankings.png")


def make_fig():
    data = {
        "Model": [
            "Dummy",
            "LR",
            "MLP",
            "LSVC",
            "RBF-SVC",
            "Poly-SVC",
            "Sig-SVC",
            "DT",
            "RF",
            "HGBC",
        ],
        "CV-ROC-AUC": [
            0.5000,
            0.9501,
            0.9790,
            0.9499,
            0.9657,
            0.9537,
            0.9473,
            0.9452,
            0.9774,
            0.9779,
        ],
        "ROC-AUC": [
            0.5000,
            0.9656,
            0.9879,
            0.9651,
            0.9796,
            0.9740,
            0.9633,
            0.9597,
            0.9885,
            0.9887,
        ],
        "Accuracy": [
            0.5000,
            0.9080,
            0.9545,
            0.9020,
            0.9090,
            0.8935,
            0.8655,
            0.9180,
            0.9455,
            0.9485,
        ],
        "Precision": [
            0.0000,
            0.9368,
            0.9586,
            0.9389,
            0.9436,
            0.9254,
            0.9518,
            0.9090,
            0.9468,
            0.9481,
        ],
        "Recall": [
            0.0000,
            0.8750,
            0.9500,
            0.8600,
            0.8700,
            0.8560,
            0.7700,
            0.9290,
            0.9440,
            0.9490,
        ],
        "F1-score": [
            0.0000,
            0.9049,
            0.9543,
            0.8977,
            0.9053,
            0.8894,
            0.8513,
            0.9189,
            0.9454,
            0.9485,
        ],
    }

    df = pd.DataFrame(data)
    df.set_index("Model", inplace=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 8), dpi=150)

    sns.heatmap(
        df,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",
        linewidths=0.5,
        cbar_kws={"label": "Score"},
    )

    plt.title(
        "Classification Performance Results (16-dimension embeddings)",
        fontsize=16,
        pad=20,
    )
    plt.yticks(rotation=0)  # Keep model names horizontal
    plt.tight_layout()
    plt.savefig("./results-fig-grid.png")

    df_melted = df.reset_index().melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )

    plt.figure(figsize=(14, 7), dpi=150)

    bar_plot = sns.barplot(
        data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis"
    )

    plt.title("Metric Comparison by Model", fontsize=16, pad=20)
    plt.ylim(0.4, 1.05)  # Zoom in on the top half (since most scores are high)
    plt.legend(
        bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0
    )  # Move legend outside
    plt.tight_layout()

    plt.savefig("./results-fig.png")


if __name__ == "__main__":
    make_fig()
    importance_rankings()
    shap_rankings()
    agg_rankings()
