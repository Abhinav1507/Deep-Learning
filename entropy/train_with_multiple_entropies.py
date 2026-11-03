
import os
import csv
import glob
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.CNN import CNN


# ============================================================
# 1. CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(
    description="CNN training with multiple weight entropy measures"
)

parser.add_argument(
    "--dataset",
    type=str,
    default="MNIST",
    choices=["MNIST", "FashionMNIST", "CIFAR10"]
)

# Total target epoch, NOT additional epochs
parser.add_argument("--epochs", type=int, default=20)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--bins", type=int, default=100)

parser.add_argument("--fixed_min", type=float, default=-1.0)
parser.add_argument("--fixed_max", type=float, default=1.0)

parser.add_argument("--output_dir", type=str, default="outputs")

# Automatic resume is the default
parser.add_argument(
    "--resume",
    type=str,
    default="auto",
    help="auto, latest, or explicit checkpoint path"
)

# Start a completely new experiment
parser.add_argument(
    "--fresh_start",
    action="store_true",
    help="Ignore existing checkpoints and start from epoch 1"
)

parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

if args.epochs < 1:
    parser.error("--epochs must be at least 1")

if args.bins < 2:
    parser.error("--bins must be at least 2")

if args.fixed_min >= args.fixed_max:
    parser.error("--fixed_min must be smaller than --fixed_max")


# ============================================================
# 2. DEVICE AND RANDOM SEEDS
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# ============================================================
# 3. OUTPUT DIRECTORIES
# ============================================================

checkpoint_dir = os.path.join(
    args.output_dir, "checkpoints"
)

entropy_dir = os.path.join(
    args.output_dir, "entropy"
)

logs_dir = os.path.join(
    args.output_dir, "logs"
)

for directory in [
    checkpoint_dir,
    entropy_dir,
    logs_dir
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# 4. DATASET
# ============================================================

if args.dataset == "CIFAR10":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        "./data",
        train=False,
        download=True,
        transform=transform
    )

    in_channels = 3

else:

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    DatasetClass = (
        datasets.MNIST
        if args.dataset == "MNIST"
        else datasets.FashionMNIST
    )

    train_dataset = DatasetClass(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = DatasetClass(
        "./data",
        train=False,
        download=True,
        transform=transform
    )

    in_channels = 1


# ============================================================
# 5. DATA LOADERS
# ============================================================

train_generator = torch.Generator().manual_seed(args.seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    generator=train_generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# ============================================================
# 6. MODEL AND OPTIMIZER
# ============================================================

model = CNN(
    num_classes=10,
    in_channels=in_channels
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr
)


# ============================================================
# 7. CSV SAVING FUNCTION
# ============================================================

def save_csv_row(filepath, fieldnames, row):

    rows = {}

    if os.path.exists(filepath):

        with open(filepath, "r", newline="") as f:

            reader = csv.DictReader(f)

            for old in reader:

                if old.get("epoch"):

                    rows[int(old["epoch"])] = {
                        key: old.get(key, "")
                        for key in fieldnames
                    }

    rows[int(row["epoch"])] = {
        key: row.get(key, "")
        for key in fieldnames
    }

    temporary_file = filepath + ".tmp"

    with open(temporary_file, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for epoch in sorted(rows):
            writer.writerow(rows[epoch])

    os.replace(temporary_file, filepath)


# ============================================================
# 8. ENTROPY FUNCTIONS
# ============================================================

def finite_numpy(weights):

    x = (
        weights.detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    return x[np.isfinite(x)]


def entropy_from_hist(hist):

    hist = np.asarray(
        hist,
        dtype=np.float64
    )

    total = hist.sum()

    if total == 0:
        return 0.0

    probabilities = hist[hist > 0] / total

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


# ------------------------------------------------------------
# 8.1 Dynamic-range histogram entropy
# ------------------------------------------------------------

def dynamic_range_histogram_entropy(weights, bins):

    """
    Original histogram entropy.

    Signed weights.
    Histogram range recalculated using current min and max.
    """

    x = finite_numpy(weights)

    if x.size == 0:
        return 0.0

    if x.min() == x.max():
        return 0.0

    hist, _ = np.histogram(
        x,
        bins=bins,
        range=(float(x.min()), float(x.max()))
    )

    return entropy_from_hist(hist)


# ------------------------------------------------------------
# 8.2 Fixed-bin histogram entropy
# ------------------------------------------------------------

def fixed_bin_histogram_entropy(weights, edges):

    """
    Signed weights.

    Same bin edges across all epochs.
    Values outside the fixed range are clipped into
    the first or last bin.
    """

    x = finite_numpy(weights)

    if x.size == 0:
        return 0.0

    x = np.clip(
        x,
        edges[0],
        edges[-1]
    )

    hist, _ = np.histogram(
        x,
        bins=edges
    )

    return entropy_from_hist(hist)


# ------------------------------------------------------------
# 8.3 Magnitude-normalised entropy
# ------------------------------------------------------------

def magnitude_normalised_entropy(weights):

    """
    p_i = |w_i| / sum(|w_j|)

    Returns:
        Raw Shannon entropy in bits.
        Normalised entropy H/log2(N).
    """

    x = np.abs(
        finite_numpy(weights)
    ).astype(np.float64, copy=False)

    n = x.size
    total = x.sum()

    if n == 0 or not np.isfinite(total) or total <= 0:
        return 0.0, 0.0

    probabilities = x / total

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    normalized_entropy = (
        entropy / np.log2(n)
        if n > 1
        else 0.0
    )

    return float(entropy), float(normalized_entropy)


# ============================================================
# 9. CALCULATE ENTROPY FOR ALL LAYERS
# ============================================================

def calculate_model_entropies(model):

    results = {}
    all_weights = []

    edges = np.linspace(
        args.fixed_min,
        args.fixed_max,
        args.bins + 1
    )

    for name, param in model.named_parameters():

        if "weight" not in name:
            continue

        weights = param.detach()

        all_weights.append(
            weights.reshape(-1)
        )

        # Dynamic-range histogram entropy
        results[
            f"dynamic_range_hist::{name}"
        ] = dynamic_range_histogram_entropy(
            weights,
            args.bins
        )

        # Fixed-bin histogram entropy
        results[
            f"fixed_bin_hist::{name}"
        ] = fixed_bin_histogram_entropy(
            weights,
            edges
        )

        # Magnitude entropy
        h, hn = magnitude_normalised_entropy(weights)

        results[
            f"magnitude_entropy_bits::{name}"
        ] = h

        results[
            f"magnitude_entropy_normalized::{name}"
        ] = hn

    if not all_weights:

        raise RuntimeError(
            "No model parameters containing 'weight' were found."
        )

    # Pool all layer weights
    pooled_weights = torch.cat(all_weights)

    # Combined dynamic-range entropy
    results[
        "dynamic_range_hist::ALL_LAYERS_COMBINED"
    ] = dynamic_range_histogram_entropy(
        pooled_weights,
        args.bins
    )

    # Combined fixed-bin entropy
    results[
        "fixed_bin_hist::ALL_LAYERS_COMBINED"
    ] = fixed_bin_histogram_entropy(
        pooled_weights,
        edges
    )

    # Combined magnitude entropy
    h, hn = magnitude_normalised_entropy(
        pooled_weights
    )

    results[
        "magnitude_entropy_bits::ALL_LAYERS_COMBINED"
    ] = h

    results[
        "magnitude_entropy_normalized::ALL_LAYERS_COMBINED"
    ] = hn

    return results


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        loss_sum += loss.item() * images.size(0)

        correct += (
            outputs.argmax(1) == labels
        ).sum().item()

        total += labels.size(0)

    return (
        loss_sum / total,
        100.0 * correct / total
    )


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)

            correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            total += labels.size(0)

    return (
        loss_sum / total,
        100.0 * correct / total
    )


# ============================================================
# 12. FIND LATEST COMPLETED CHECKPOINT
# ============================================================

def find_latest_checkpoint():

    pattern = os.path.join(
        checkpoint_dir,
        "model_epoch_*.pth"
    )

    paths = glob.glob(pattern)

    valid_checkpoints = []

    for path in paths:

        filename = os.path.basename(path)

        try:

            epoch_number = int(
                filename.replace(
                    "model_epoch_", ""
                ).replace(
                    ".pth", ""
                )
            )

            valid_checkpoints.append(
                (epoch_number, path)
            )

        except ValueError:

            continue

    if not valid_checkpoints:
        return None

    valid_checkpoints.sort(
        key=lambda item: item[0]
    )

    return valid_checkpoints[-1]


# ============================================================
# 13. AUTOMATIC RESUME
# ============================================================

start_epoch = 1
latest_epoch = 0
resume_path = None

if args.fresh_start:

    print("\nFresh start requested.")
    print("Existing checkpoints will be ignored.")

else:

    if args.resume.lower() in ["auto", "latest"]:

        latest = find_latest_checkpoint()

        if latest is not None:

            latest_epoch, resume_path = latest

    else:

        resume_path = args.resume

    if resume_path is not None:

        if not os.path.isfile(resume_path):

            raise FileNotFoundError(
                f"Checkpoint not found: {resume_path}"
            )

        print("\nLoading checkpoint:")
        print(resume_path)

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False
        )

        if (
            "model_state_dict" not in checkpoint
            or "optimizer_state_dict" not in checkpoint
            or "epoch" not in checkpoint
        ):

            raise ValueError(
                "The selected file is not a full training checkpoint. "
                "Use model_epoch_XXX.pth, not model_weights_epoch_XXX.pth."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        latest_epoch = int(checkpoint["epoch"])

        start_epoch = latest_epoch + 1

        # Restore random-number-generator states
        if "rng_state" in checkpoint:

            rng = checkpoint["rng_state"]

            random.setstate(rng["python"])

            np.random.set_state(rng["numpy"])

            torch.set_rng_state(rng["torch"])

            if (
                torch.cuda.is_available()
                and rng.get("cuda") is not None
            ):

                torch.cuda.set_rng_state_all(
                    rng["cuda"]
                )

        # Restore DataLoader shuffle generator
        if "loader_generator_state" in checkpoint:

            train_generator.set_state(
                checkpoint["loader_generator_state"]
            )

        # Detect dataset transition
        old_dataset = checkpoint.get(
            "config", {}
        ).get("dataset")

        if (
            old_dataset is not None
            and old_dataset != args.dataset
        ):

            print("\nDataset transition detected.")
            print("Previous dataset:", old_dataset)
            print("New dataset:", args.dataset)
            print("Model and optimizer states will be continued.")

        # Warn if important training settings changed
        old_config = checkpoint.get("config", {})

        if old_config:

            for setting in [
                "batch_size",
                "lr",
                "bins",
                "fixed_min",
                "fixed_max"
            ]:

                old_value = old_config.get(setting)
                new_value = getattr(args, setting)

                if (
                    old_value is not None
                    and old_value != new_value
                ):

                    print(
                        f"WARNING: {setting} changed from "
                        f"{old_value} to {new_value}"
                    )

        print("\nCheckpoint successfully restored.")
        print("Last completed epoch:", latest_epoch)
        print("Next epoch:", start_epoch)

    else:

        print("\nNo previous checkpoint found.")
        print("Starting from epoch 1.")


# ============================================================
# 14. CHECK TARGET EPOCH
# ============================================================

print("\n" + "=" * 60)
print("TRAINING PLAN")
print("=" * 60)

print("Dataset:", args.dataset)
print("Last completed epoch:", latest_epoch)
print("Starting epoch:", start_epoch)
print("Target epoch:", args.epochs)

if start_epoch > args.epochs:

    print("\nRequested target epoch already completed.")
    print("No additional training is required.")

    raise SystemExit(0)

remaining_epochs = args.epochs - start_epoch + 1

print("Remaining epochs:", remaining_epochs)

print("=" * 60)


# ============================================================
# 15. CSV FILES
# ============================================================

entropy_csv = os.path.join(
    entropy_dir,
    "weight_entropy.csv"
)

training_csv = os.path.join(
    logs_dir,
    "training_log.csv"
)

accuracy_csv = os.path.join(
    logs_dir,
    "accuracy_history.csv"
)


# ============================================================
# 16. MAIN TRAINING LOOP
# ============================================================

for epoch in range(start_epoch, args.epochs + 1):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch}/{args.epochs}")
    print("=" * 60)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_acc = train_one_epoch()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    test_loss, test_acc = evaluate()

    accuracy_gap = train_acc - test_acc
    loss_gap = test_loss - train_loss

    # --------------------------------------------------------
    # Calculate entropy
    # --------------------------------------------------------

    entropies = calculate_model_entropies(model)

    print("\nCombined entropy:")

    for metric in [
        "dynamic_range_hist",
        "fixed_bin_hist",
        "magnitude_entropy_bits",
        "magnitude_entropy_normalized"
    ]:

        key = f"{metric}::ALL_LAYERS_COMBINED"

        print(
            f"{metric}: {entropies[key]:.6f}"
        )

    # --------------------------------------------------------
    # Save entropy CSV
    # --------------------------------------------------------

    save_csv_row(
        entropy_csv,
        ["epoch"] + list(entropies.keys()),
        {
            "epoch": epoch,
            **entropies
        }
    )

    # --------------------------------------------------------
    # Save accuracy CSV
    # --------------------------------------------------------

    accuracy_row = {
        "epoch": epoch,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "accuracy_gap": accuracy_gap,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "loss_gap": loss_gap
    }

    save_csv_row(
        accuracy_csv,
        list(accuracy_row.keys()),
        accuracy_row
    )

    # --------------------------------------------------------
    # Save training CSV
    # --------------------------------------------------------

    training_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "accuracy_gap": accuracy_gap,
        "loss_gap": loss_gap,

        "combined_dynamic_range_entropy":
            entropies[
                "dynamic_range_hist::ALL_LAYERS_COMBINED"
            ],

        "combined_fixed_bin_entropy":
            entropies[
                "fixed_bin_hist::ALL_LAYERS_COMBINED"
            ],

        "combined_magnitude_entropy_bits":
            entropies[
                "magnitude_entropy_bits::ALL_LAYERS_COMBINED"
            ],

        "combined_magnitude_entropy_normalized":
            entropies[
                "magnitude_entropy_normalized::ALL_LAYERS_COMBINED"
            ]
    }

    save_csv_row(
        training_csv,
        list(training_row.keys()),
        training_row
    )

    # ========================================================
    # 17. SAVE FULL RESUMABLE CHECKPOINT
    # ========================================================

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "train_loss": train_loss,
        "train_accuracy": train_acc,

        "test_loss": test_loss,
        "test_accuracy": test_acc,

        "accuracy_gap": accuracy_gap,
        "loss_gap": loss_gap,

        "entropy": entropies,

        "config": vars(args),

        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),

            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
        },

        "loader_generator_state": (
            train_generator.get_state()
        )
    }

    # Save one full checkpoint for every epoch
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"model_epoch_{epoch:03d}.pth"
    )

    temporary_checkpoint = checkpoint_path + ".tmp"

    torch.save(
        checkpoint,
        temporary_checkpoint
    )

    os.replace(
        temporary_checkpoint,
        checkpoint_path
    )

    # ========================================================
    # 18. SAVE WEIGHTS-ONLY CHECKPOINT
    # ========================================================

    weights_path = os.path.join(
        checkpoint_dir,
        f"model_weights_epoch_{epoch:03d}.pth"
    )

    temporary_weights = weights_path + ".tmp"

    torch.save(
        model.state_dict(),
        temporary_weights
    )

    os.replace(
        temporary_weights,
        weights_path
    )

    # ========================================================
    # 19. UPDATE LATEST FULL CHECKPOINT
    # ========================================================

    latest_path = os.path.join(
        checkpoint_dir,
        "latest_checkpoint.pth"
    )

    temporary_latest = latest_path + ".tmp"

    torch.save(
        checkpoint,
        temporary_latest
    )

    os.replace(
        temporary_latest,
        latest_path
    )

    # ========================================================
    # 20. PRINT EPOCH SUMMARY
    # ========================================================

    print("\nTraining results:")

    print(f"Train loss: {train_loss:.5f}")
    print(f"Train accuracy: {train_acc:.2f}%")

    print(f"Test loss: {test_loss:.5f}")
    print(f"Test accuracy: {test_acc:.2f}%")

    print("\nSaved checkpoints:")

    print(checkpoint_path)
    print(weights_path)
    print(latest_path)

    print("\nUpdated CSV files:")

    print(entropy_csv)
    print(training_csv)
    print(accuracy_csv)


# ============================================================
# 21. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SESSION COMPLETED")
print("=" * 60)

print("Dataset:", args.dataset)
print("Target epoch:", args.epochs)

latest = find_latest_checkpoint()

if latest is not None:
    print("Latest saved epoch:", latest[0])

print("Output directory:", args.output_dir)
