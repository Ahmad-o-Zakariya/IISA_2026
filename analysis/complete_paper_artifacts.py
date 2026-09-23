"""Create reproducible F1, scalogram-inspection, and Grad-CAM paper artifacts."""
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
GRADCAM = RESULTS / "gradcam"
MODELS = ROOT / "models"
CLASSES = ["ball", "combined", "healthy", "inner_race", "looseness", "outer_race"]
SEED = 42


def image_array(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def make_f1_summary():
    within = []
    cross = []
    for fold in range(1, 6):
        report = pd.read_csv(RESULTS / f"within_speed_fold_{fold}_classification_report.csv", index_col=0)
        within.append(report.loc[CLASSES, "f1-score"].rename(fold))
    for fold in range(1, 5):
        report = pd.read_csv(RESULTS / f"cross_speed_fold_{fold}_classification_report.csv", index_col=0)
        cross.append(report.loc[CLASSES, "f1-score"].rename(fold))
    within_df = pd.concat(within, axis=1)
    cross_df = pd.concat(cross, axis=1)
    output = pd.DataFrame({
        "Fault Class": CLASSES,
        "Within-Speed F1 Mean (across 5 folds)": within_df.mean(axis=1).reindex(CLASSES).values,
        "Within-Speed F1 Std (across 5 folds)": within_df.std(axis=1).reindex(CLASSES).values,
        "Cross-Speed F1 (mean across 4 folds; 642+1236 trained, 1809 tested)": cross_df.mean(axis=1).reindex(CLASSES).values,
    })
    output["Change (Cross minus Within)"] = (output.iloc[:, 3] - output.iloc[:, 1])
    output.to_csv(RESULTS / "per_class_f1_summary.csv", index=False)
    return output


def make_scalogram_figures(meta):
    rng = np.random.default_rng(SEED)
    selected = {}
    for cls in ["healthy", "outer_race"]:
        for speed in ["642", "1236", "1809"]:
            candidates = meta[(meta.fault == cls) & (meta.speed.astype(str) == speed)]
            selected[cls, speed] = candidates.iloc[rng.choice(len(candidates), 4, replace=False)]

    fig, axes = plt.subplots(2, 12, figsize=(24, 4.8), constrained_layout=True)
    for row, cls in enumerate(["healthy", "outer_race"]):
        for speed_idx, speed in enumerate(["642", "1236", "1809"]):
            for sample_idx, (_, item) in enumerate(selected[cls, speed].iterrows()):
                ax = axes[row, speed_idx * 4 + sample_idx]
                ax.imshow(image_array(item.image_path), cmap="gray", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0 and sample_idx == 1:
                    ax.set_title(f"{speed} RPM", fontsize=11)
        axes[row, 0].set_ylabel(cls.replace("_", " "), fontsize=12)
    fig.savefig(FIGURES / "healthy_vs_outerrace_comparison.png", dpi=300)
    plt.close(fig)

    means = {}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for row, cls in enumerate(["healthy", "outer_race"]):
        for col, speed in enumerate(["642", "1236", "1809"]):
            paths = meta[(meta.fault == cls) & (meta.speed.astype(str) == speed)].image_path
            avg = np.mean([image_array(p) for p in paths], axis=0)
            means[cls, speed] = avg
            ax = axes[row, col]
            ax.imshow(avg, cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"{cls.replace('_', ' ')} - {speed} RPM")
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(FIGURES / "healthy_vs_outerrace_average.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    stats = []
    for ax, speed in zip(axes, ["642", "1236", "1809"]):
        diff = np.abs(means["healthy", speed] - means["outer_race", speed])
        ax.imshow(diff, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Absolute difference - {speed} RPM")
        ax.set_xticks([]); ax.set_yticks([])
        thirds = np.array_split(diff, 3, axis=0)
        stats.append({"speed": speed, "high_frequency_image_rows": float(thirds[0].mean()),
                      "mid_frequency_image_rows": float(thirds[1].mean()),
                      "low_frequency_image_rows": float(thirds[2].mean()), "overall": float(diff.mean())})
    fig.savefig(FIGURES / "healthy_vs_outerrace_difference.png", dpi=300)
    plt.close(fig)
    pd.DataFrame(stats).to_csv(RESULTS / "healthy_vs_outerrace_pixel_difference_stats.csv", index=False)


def load_model():
    checkpoint = torch.load(MODELS / "best_resnet18_cross_speed_fold_4.pth", map_location="cpu", weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, len(CLASSES)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["class_to_idx"]


NORMALIZE = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def cam_for(model, path, target_index):
    activation = []
    gradient = []
    handle_f = model.layer4.register_forward_hook(lambda m, i, o: activation.append(o.detach()))
    handle_b = model.layer4.register_full_backward_hook(lambda m, gi, go: gradient.append(go[0].detach()))
    image = Image.open(path).convert("RGB")
    x = NORMALIZE(image).unsqueeze(0)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    logits[0, target_index].backward()
    weights = gradient[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation[0]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    handle_f.remove(); handle_b.remove()
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0, cam


def overlay(original, cam):
    return 0.45 * np.dstack([original] * 3) + 0.55 * plt.get_cmap("jet")(cam)[..., :3]


def make_gradcam(meta):
    GRADCAM.mkdir(parents=True, exist_ok=True)
    model, class_to_idx = load_model()
    pred = pd.read_csv(RESULTS / "cross_speed_fold_4_predictions.csv")
    test = meta.merge(pred[["image_path", "true_label", "pred_label", "correct"]], on="image_path", how="inner")
    if len(test) != len(pred):
        raise RuntimeError("Prediction metadata does not match scalogram metadata.")

    fig, axes = plt.subplots(6, 3, figsize=(10, 17), constrained_layout=True)
    for row, cls in enumerate(CLASSES):
        rows = test[(test.true_label == cls) & test.correct].head(3)
        if len(rows) < 3:
            raise RuntimeError(f"Fewer than three correct {cls} examples are available.")
        for col, (_, item) in enumerate(rows.iterrows()):
            original, cam = cam_for(model, item.image_path, class_to_idx[cls])
            axes[row, col].imshow(overlay(original, cam), aspect="auto")
            axes[row, col].set_title(f"{cls} | {item.speed} RPM", fontsize=9)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.savefig(GRADCAM / "gradcam_correct_predictions.png", dpi=300)
    plt.close(fig)

    confused = pd.concat([
        test[(test.true_label == "healthy") & (test.pred_label == "outer_race")].head(4),
        test[(test.true_label == "outer_race") & (test.pred_label == "healthy")].head(4),
    ], ignore_index=True)
    if confused.empty:
        raise RuntimeError("No healthy/outer_race confusions in cross-speed fold 4.")
    fig, axes = plt.subplots(len(confused), 3, figsize=(10, 3.2 * len(confused)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, (_, item) in enumerate(confused.iterrows()):
        original, cam = cam_for(model, item.image_path, class_to_idx[item.pred_label])
        for ax, data, title in zip(axes[row], [original, cam, overlay(original, cam)], ["Original", "Grad-CAM", "Overlay"]):
            ax.imshow(data, cmap="gray" if data.ndim == 2 else None, aspect="auto")
            ax.set_title(f"{title}\ntrue={item.true_label}; pred={item.pred_label}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(GRADCAM / "gradcam_healthy_outerrace_confusion.png", dpi=300)
    plt.close(fig)

    examples = pd.concat([test[(test.true_label == cls) & test.correct].head(3) for cls in ["healthy", "outer_race"]], ignore_index=True)
    fig, axes = plt.subplots(6, 2, figsize=(8, 16), constrained_layout=True)
    for row, (_, item) in enumerate(examples.iterrows()):
        wrong = "outer_race" if item.true_label == "healthy" else "healthy"
        original, correct_cam = cam_for(model, item.image_path, class_to_idx[item.true_label])
        _, wrong_cam = cam_for(model, item.image_path, class_to_idx[wrong])
        for ax, cam, title in zip(axes[row], [correct_cam, wrong_cam], [f"correct logit: {item.true_label}", f"confused logit: {wrong}"]):
            ax.imshow(overlay(original, cam), aspect="auto")
            ax.set_title(f"true={item.true_label}; {title}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(GRADCAM / "gradcam_healthy_outerrace_comparison.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    meta = pd.read_csv(RESULTS / "scalograms_metadata.csv")
    make_f1_summary()
    make_scalogram_figures(meta)
    make_gradcam(meta)
