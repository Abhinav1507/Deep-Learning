
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



# ==================================================
# 1. CONFIGURATION
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="MNIST",
                    choices=["MNIST", "FashionMNIST", "CIFAR10"])

parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--bins", type=int, default=100)

parser.add_argument("--output_dir", type=str, default="outputs")

# Resume options:
# None       = Start new training
# latest     = Resume latest checkpoint
# Path       = Resume specified checkpoint

parser.add_argument("--resume", type=str, default=None)

parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()


# ==================================================
# 2. DEVICE AND RANDOM SEED
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# ==================================================
# 3. DIRECTORIES
# ==================================================

checkpoint_dir = os.path.join(
    args.output_dir, "checkpoints"
)

entropy_dir = os.path.join(
    args.output_dir, "entropy"
)

logs_dir = os.path.join(
    args.output_dir, "logs"
)

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(entropy_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)


# ==================================================
# 4. DATASET LOADING
# ==================================================

if args.dataset == "CIFAR10":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    in_channels = 3
    num_classes = 10

else:

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    DatasetClass = (
        datasets.MNIST
        if args.dataset == "MNIST"
        else datasets.FashionMNIST
    )

    train_dataset = DatasetClass(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = DatasetClass(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    in_channels = 1
    num_classes = 10


# Dedicated generator allows its RNG state to be checkpointed.
loader_generator = torch.Generator()
loader_generator.manual_seed(args.seed)


train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    generator=loader_generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# ==================================================
# 5. MODEL AND OPTIMIZER
# ==================================================

model = CNN(
    num_classes=num_classes,
    in_channels=in_channels
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr
)


# ==================================================
# 6. CSV UPSERT FUNCTION
# ==================================================

# Replaces an existing epoch row instead of creating
# duplicate rows when resuming an older checkpoint.

def save_csv_row(filepath, fieldnames, row):

    rows = {}

    if os.path.exists(filepath):

        with open(filepath, "r", newline="") as f:

            reader = csv.DictReader(f)

            for existing_row in reader:

                rows[int(existing_row["epoch"])] = existing_row

    epoch = int(row["epoch"])

    rows[epoch] = {
        key: row.get(key, "")
        for key in fieldnames
    }

    temp_path = filepath + ".tmp"

    with open(temp_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for current_epoch in sorted(rows):

            writer.writerow(rows[current_epoch])

    os.replace(temp_path, filepath)


# ==================================================
# 7. SHANNON WEIGHT ENTROPY
# ==================================================

def calculate_entropy(weights, bins=100):

    weights = weights.detach().float().cpu().numpy()
    weights = weights.flatten()

    weights = weights[np.isfinite(weights)]

    if weights.size == 0:
        return 0.0

    if np.min(weights) == np.max(weights):
        return 0.0

    hist, _ = np.histogram(
        weights,
        bins=bins,
        range=(np.min(weights), np.max(weights))
    )

    probabilities = hist / hist.sum()

    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


# ==================================================
# 8. LAYER-WISE AND COMBINED ENTROPY
# ==================================================

def calculate_model_entropy(model, bins=100):

    entropy_results = {}

    all_weights = []

    for name, param in model.named_parameters():

        if "weight" not in name:
            continue

        layer_weights = param.detach()

        entropy_results[name] = calculate_entropy(
            layer_weights,
            bins=bins
        )

        all_weights.append(
            layer_weights.reshape(-1)
        )

    combined_weights = torch.cat(all_weights)

    entropy_results["ALL_LAYERS_COMBINED"] = (
        calculate_entropy(
            combined_weights,
            bins=bins
        )
    )

    return entropy_results


# ==================================================
# 9. TRAINING FUNCTION
# ==================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / total

    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ==================================================
# 10. TESTING FUNCTION
# ==================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / total

    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ==================================================
# 11. CHECKPOINT RESUME
# ==================================================

start_epoch = 1

if args.resume is not None:

    if args.resume.lower() == "latest":

        checkpoints = glob.glob(
            os.path.join(
                checkpoint_dir,
                "model_epoch_*.pth"
            )
        )

        if not checkpoints:
            raise FileNotFoundError(
                "No checkpoints found for resuming."
            )

        resume_path = max(
            checkpoints,
            key=lambda p: int(
                os.path.basename(p).split("_")[-1].split(".")[0]
            )
        )

    else:

        resume_path = args.resume

    print("\nLoading checkpoint:", resume_path)

    # Only load checkpoints that you trust.
    checkpoint = torch.load(
        resume_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    # Restore random number generator states.
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

    if "loader_generator_state" in checkpoint:

        loader_generator.set_state(
            checkpoint["loader_generator_state"]
        )

    print("Resuming from epoch:", start_epoch)

    print(
        "Previous training accuracy:",
        checkpoint.get("train_accuracy", "N/A")
    )

    print(
        "Previous test accuracy:",
        checkpoint.get("test_accuracy", "N/A")
    )


# ==================================================
# 12. CSV FILE PATHS
# ==================================================

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


# ==================================================
# 13. MAIN TRAINING LOOP
# ==================================================

for epoch in range(start_epoch, args.epochs + 1):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch}/{args.epochs}")
    print("=" * 60)

    # ----------------------------------------------
    # TRAIN
    # ----------------------------------------------

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    # ----------------------------------------------
    # TEST
    # ----------------------------------------------

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    # ----------------------------------------------
    # ACCURACY AND GENERALIZATION ANALYSIS
    # ----------------------------------------------

    accuracy_gap = train_acc - test_acc

    loss_gap = test_loss - train_loss

    print("\nTraining Results")

    print(f"Train Loss     : {train_loss:.6f}")
    print(f"Train Accuracy : {train_acc:.4f}%")

    print("\nTesting Results")

    print(f"Test Loss      : {test_loss:.6f}")
    print(f"Test Accuracy  : {test_acc:.4f}%")

    print("\nGeneralization Analysis")

    print(f"Accuracy Gap   : {accuracy_gap:.4f}%")
    print(f"Loss Gap       : {loss_gap:.6f}")

    # ----------------------------------------------
    # CALCULATE ENTROPY
    # ----------------------------------------------

    entropy_results = calculate_model_entropy(
        model,
        bins=args.bins
    )

    print("\nWeight Entropy")

    for layer, entropy in entropy_results.items():

        print(f"{layer}: {entropy:.6f} bits")

    # ----------------------------------------------
    # SAVE ENTROPY CSV
    # ----------------------------------------------

    entropy_row = {
        "epoch": epoch,
        **entropy_results
    }

    entropy_fields = [
        "epoch"
    ] + list(entropy_results.keys())

    save_csv_row(
        entropy_csv,
        entropy_fields,
        entropy_row
    )

    # ----------------------------------------------
    # SAVE ACCURACY HISTORY
    # ----------------------------------------------

    accuracy_row = {

        "epoch": epoch,

        "train_accuracy": train_acc,

        "test_accuracy": test_acc,

        "accuracy_gap": accuracy_gap,

        "train_loss": train_loss,

        "test_loss": test_loss,

        "loss_gap": loss_gap
    }

    accuracy_fields = [
        "epoch",
        "train_accuracy",
        "test_accuracy",
        "accuracy_gap",
        "train_loss",
        "test_loss",
        "loss_gap"
    ]

    save_csv_row(
        accuracy_csv,
        accuracy_fields,
        accuracy_row
    )

    # ----------------------------------------------
    # SAVE TRAINING LOG
    # ----------------------------------------------

    training_row = {

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy": train_acc,

        "test_loss": test_loss,

        "test_accuracy": test_acc,

        "accuracy_gap": accuracy_gap,

        "loss_gap": loss_gap,

        "combined_weight_entropy":
            entropy_results["ALL_LAYERS_COMBINED"]
    }

    training_fields = list(training_row.keys())

    save_csv_row(
        training_csv,
        training_fields,
        training_row
    )

    # ----------------------------------------------
    # SAVE COMPLETE CHECKPOINT
    # ----------------------------------------------

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"model_epoch_{epoch:03d}.pth"
    )

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

        "entropy": entropy_results,

        "config": vars(args),

        # Save random states for reproducible continuation.
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

        "loader_generator_state":
            loader_generator.get_state()
    }

    # Atomic save: avoid leaving a partially written checkpoint.
    temp_checkpoint = checkpoint_path + ".tmp"

    torch.save(
        checkpoint,
        temp_checkpoint
    )

    os.replace(
        temp_checkpoint,
        checkpoint_path
    )

    print("\nCheckpoint saved:", checkpoint_path)

    print("Entropy history saved.")
    print("Accuracy history saved.")
    print("Training history saved.")

    print("\nEpoch completed successfully.")


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("All checkpoints, entropy and accuracy records saved.")
#python entropy/train.py --dataset MNIST --epochs 20
#PYTHONPATH=. python entropy/train.py --dataset MNIST --epochs 20
""" till 90 epoch I used same MNIST dataset , 
 from 91 I am going to use CIFAR10 dataset"""
#PYTHONPATH=. python entropy/train.py --dataset FashionMNIST --epochs 180 --resume latest