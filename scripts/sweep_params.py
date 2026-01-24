import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluate_detect import main as eval_main


def run_grid(
    split="valid",
    score_thrs=(1.5, 1.7, 1.9, 2.0, 2.1),
    nms_thrs=(0.4, 0.2, 0.1),
    iou_thr=0.5,
):
    rows = []
    for nms in nms_thrs:
        for s in score_thrs:
            metrics = eval_main(split, iou_thr=iou_thr, nms_thr=nms, score_thr=s)
            rows.append({
                "split": split,
                "score_thr": s,
                "nms_thr": nms,
                **metrics
            })

    df = pd.DataFrame(rows).sort_values(["nms_thr", "score_thr"])
    df.to_csv("sweep_results.csv", index=False)
    print("Saved sweep_results.csv")
    return df


def plot_f1_vs_score(df):
    plt.figure()
    for nms, g in df.groupby("nms_thr"):
        plt.plot(g["score_thr"], g["F1"], marker="o", label=f"NMS={nms}")
    plt.xlabel("score_thr")
    plt.ylabel("F1")
    plt.title("Wpływ score_thr i NMS na F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_f1.png", dpi=200)
    print("Saved plot_f1.png")


def plot_precision_recall(df):
    plt.figure()
    for nms, g in df.groupby("nms_thr"):
        plt.plot(g["score_thr"], g["Precision"], marker="o", label=f"Precision, NMS={nms}")
    plt.xlabel("score_thr")
    plt.ylabel("Precision")
    plt.title("Precision vs score_thr")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_precision.png", dpi=200)
    print("Saved plot_precision.png")

    plt.figure()
    for nms, g in df.groupby("nms_thr"):
        plt.plot(g["score_thr"], g["Recall"], marker="o", label=f"Recall, NMS={nms}")
    plt.xlabel("score_thr")
    plt.ylabel("Recall")
    plt.title("Recall vs score_thr")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_recall.png", dpi=200)
    print("Saved plot_recall.png")


if __name__ == "__main__":
    df = run_grid()
    plot_f1_vs_score(df)
    plot_precision_recall(df)
