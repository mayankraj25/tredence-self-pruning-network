"""
Self-Pruning Neural Network on CIFAR-10
Author: Mayank Raj

Each linear layer gets a learnable gate per weight. Gates are sigmoid-activated,
so they live in (0,1) and get multiplied element-wise with the weights before
any computation. Throwing an L1 penalty on those gate values during training
pushes most of them to zero — which is the pruning. No post-training surgery
needed, the network figures out what to drop on its own.
"""

import os
import time
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display on headless machines
import matplotlib.pyplot as plt


class PrunableLinear(nn.Module):
    """
    Same interface as nn.Linear but with an extra gate_scores parameter
    that has the same shape as the weight matrix.

    The forward pass:
        gates        = sigmoid(gate_scores)   -> values in (0, 1)
        pruned_w     = weight * gates         -> element-wise
        out          = x @ pruned_w.T + bias

    Both weight and gate_scores are registered as Parameters, so autograd
    handles gradients for both without any extra work on our side.

    weight      : (out_features, in_features)
    bias        : (out_features,)
    gate_scores : (out_features, in_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # init at 0.5 — middle of the (0,1) range so the L1 penalty
        # and classification loss start on equal footing
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self):
        # matching nn.Linear's default init exactly
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.constant_(self.gate_scores, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # clamp to (0,1) instead of sigmoid — gradient is 1.0 everywhere
        # in range, so L1 pressure transmits without attenuation
        gates = self.gate_scores.clamp(0.0, 1.0)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        """fraction of gates that have effectively closed"""
        gates = self.gate_scores.clamp(0.0, 1.0)
        pruned = (gates < threshold).float().mean().item()
        return pruned

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


class SelfPruningNet(nn.Module):
    """
    Four-layer MLP for CIFAR-10. Every linear layer is PrunableLinear.

    Flatten -> PrunableLinear(3072->1024) -> BN -> ReLU -> Dropout
            -> PrunableLinear(1024->512)  -> BN -> ReLU -> Dropout
            -> PrunableLinear(512->256)   -> BN -> ReLU -> Dropout
            -> PrunableLinear(256->10)

    BN before activation helps keep training stable once gates start
    zeroing out chunks of the weight matrix mid-training.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            PrunableLinear(3 * 32 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def prunable_layers(self):
        """yields every PrunableLinear in the network"""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm across all gate values in the network.

        L1 works here because it applies a constant gradient regardless of
        how small the gate already is — unlike L2 which eases off near zero.
        That constant pull is what actually gets gates to land on exactly 0
        rather than just getting close. Since sigmoid output is always >= 0,
        abs() is technically redundant but makes the intent explicit.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for layer in self.prunable_layers():
            gates = layer.gate_scores.clamp(0.0, 1.0)
            total = total + gates.sum()
            count += gates.numel()
        return total / count

    @torch.no_grad()
    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """percentage of weights whose gate has dropped below the threshold"""
        pruned_count = 0
        total_count  = 0
        for layer in self.prunable_layers():
            gates = layer.gate_scores.clamp(0.0, 1.0)
            pruned_count += (gates < threshold).sum().item()
            total_count  += gates.numel()
        return pruned_count / total_count if total_count > 0 else 0.0

    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        """flat array of every gate value — used for the distribution plot"""
        vals = []
        for layer in self.prunable_layers():
            gates = layer.gate_scores.clamp(0.0, 1.0)
            vals.append(gates.detach().cpu().numpy().ravel())
        return np.concatenate(vals)


def get_cifar10_loaders(data_dir: str = "./data", batch_size: int = 256,
                        num_workers: int = 0):
    """
    Standard CIFAR-10 setup with typical augmentations for train.
    Channel stats are the usual CIFAR-10 values.

    num_workers and pin_memory are handled carefully here — MPS on macOS
    deadlocks with multiprocessing workers, and pin_memory is a CUDA thing
    that just adds overhead on Apple Silicon.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    is_mps = torch.backends.mps.is_available()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    pin = not is_mps
    nw  = 0 if is_mps else num_workers

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=nw,
                              pin_memory=pin, persistent_workers=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=nw,
                              pin_memory=pin, persistent_workers=False)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, lambda_sparse, device, scaler=None):
    """
    One full pass over the training set.
    Total loss = CrossEntropy + lambda * L1_gate_penalty

    returns: (avg_ce, avg_sparsity_loss, avg_total_loss, accuracy)
    """
    model.train()
    total_ce      = 0.0
    total_sparse  = 0.0
    total_loss_v  = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        dt = device.type
        if dt == "cuda" and scaler is not None:
            # full AMP on CUDA
            with torch.amp.autocast("cuda"):
                logits        = model(images)
                ce_loss       = F.cross_entropy(logits, labels)
                sparse_loss   = model.sparsity_loss()
                total_loss    = ce_loss + lambda_sparse * sparse_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif dt == "mps":
            # MPS supports autocast but not GradScaler
            with torch.amp.autocast("mps"):
                logits        = model(images)
                ce_loss       = F.cross_entropy(logits, labels)
                sparse_loss   = model.sparsity_loss()
                total_loss    = ce_loss + lambda_sparse * sparse_loss
            total_loss.backward()
            optimizer.step()
        else:
            logits       = model(images)
            ce_loss      = F.cross_entropy(logits, labels)
            sparse_loss  = model.sparsity_loss()
            total_loss   = ce_loss + lambda_sparse * sparse_loss
            total_loss.backward()
            optimizer.step()

        total_ce     += ce_loss.item()
        total_sparse += sparse_loss.item()
        total_loss_v += total_loss.item()

        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    return (total_ce / n, total_sparse / n, total_loss_v / n, correct / total)


@torch.no_grad()
def evaluate(model, loader, device):
    """test accuracy"""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits  = model(images)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def run_experiment(lambda_sparse: float,
                   train_loader, test_loader,
                   device,
                   epochs: int = 30,
                   lr: float = 1e-3,
                   dropout_rate: float = 0.3,
                   seed: int = 42,
                   save_dir: str = "./outputs"):
    """
    Trains a model for a given lambda and returns results + gate values.

    Saves the best checkpoint (by test acc) and reloads it at the end
    so the returned metrics reflect the actual best model, not last epoch.

    args:
        lambda_sparse : weight on the sparsity term
        train_loader  : training DataLoader
        test_loader   : test DataLoader
        device        : torch device
        epochs        : training epochs
        lr            : Adam learning rate
        dropout_rate  : dropout in the network
        seed          : for reproducibility
        save_dir      : where to write checkpoints

    returns dict: lambda, test_acc, sparsity, gate_values, history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = SelfPruningNet(dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # cosine decay avoids the lr bouncing around at the end of training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    # GradScaler only on CUDA; MPS uses autocast without it
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"ce": [], "sparse": [], "total": [], "train_acc": [], "test_acc": []}

    print(f"\n{'='*60}")
    print(f"  λ = {lambda_sparse}")
    print(f"  Device : {device}  |  Epochs : {epochs}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6} {'CE':>8} {'Sparse':>10} {'Total':>10} "
          f"{'Tr Acc':>8} {'Te Acc':>8} {'Sparsity':>10} {'LR':>10}")
    print("-" * 78)

    best_test_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ce, sparse, total, tr_acc = train_epoch(
            model, train_loader, optimizer, lambda_sparse, device, scaler)
        te_acc   = evaluate(model, test_loader, device)
        sparsity = model.overall_sparsity()
        lr_now   = scheduler.get_last_lr()[0]

        history["ce"].append(ce)
        history["sparse"].append(sparse)
        history["total"].append(total)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        scheduler.step()

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"best_lambda_{lambda_sparse}.pt"))

        print(f"{epoch:>6} {ce:>8.4f} {sparse:>10.4f} {total:>10.4f} "
              f"{tr_acc:>8.4%} {te_acc:>8.4%} {sparsity:>10.4%} {lr_now:>10.2e}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  |  Best test acc: {best_test_acc:.4%}  "
          f"|  Final sparsity: {sparsity:.4%}")

    # reload best checkpoint for final evaluation
    best_state = torch.load(
        os.path.join(save_dir, f"best_lambda_{lambda_sparse}.pt"),
        map_location=device)
    model.load_state_dict(best_state)
    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity()
    gate_values    = model.all_gate_values()

    return {
        "lambda"      : lambda_sparse,
        "test_acc"    : final_test_acc,
        "sparsity"    : final_sparsity,
        "gate_values" : gate_values,
        "history"     : history,
    }


def plot_gate_distribution(gate_values: np.ndarray,
                           lambda_val: float,
                           save_path: str):
    """
    Histogram of gate values after training. A working run should show
    a big spike near 0 (pruned) and a smaller cluster toward 1 (kept).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(gate_values, bins=100, color="#3A86FF", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Gate Value Distribution  |  λ = {lambda_val}", fontsize=14)
    ax.set_xlim(-0.02, 1.02)

    near_zero = np.mean(gate_values < 0.01) * 100
    ax.axvline(0.01, color="#FF6B6B", linestyle="--", linewidth=1.2,
               label=f"Threshold = 0.01")
    ax.text(0.03, ax.get_ylim()[1] * 0.85,
            f"{near_zero:.1f}% pruned", color="#FF6B6B", fontsize=11)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot saved] {save_path}")


def plot_training_curves(results: list, save_path: str):
    """accuracy and sparsity loss over epochs for all lambda values"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours   = ["#3A86FF", "#FF6B6B", "#8AC926"]

    for i, res in enumerate(results):
        lam     = res["lambda"]
        history = res["history"]
        epochs  = range(1, len(history["test_acc"]) + 1)
        colour  = colours[i % len(colours)]

        axes[0].plot(epochs, [a * 100 for a in history["test_acc"]],
                     label=f"λ={lam}", color=colour, linewidth=1.8)
        axes[1].plot(epochs, history["sparse"],
                     label=f"λ={lam}", color=colour, linewidth=1.8)

    axes[0].set_title("Test Accuracy vs Epoch",  fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title("Sparsity Loss vs Epoch", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Sparsity Loss (L1 sum)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle("Self-Pruning Network – Training Dynamics", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot saved] {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network on CIFAR-10")
    parser.add_argument("--lambdas",   nargs="+", type=float,
                        default=[1e-4, 5e-4, 2e-3],
                        help="Sparsity regularisation coefficients to sweep")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=256)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--data_dir",  type=str,   default="./data")
    parser.add_argument("--out_dir",   type=str,   default="./outputs")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--workers",   type=int,   default=2)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\nUsing device: {device}")
    if device.type == "mps":
        print("  Apple Silicon detected — MPS backend active.")
        print("  DataLoader workers=0, pin_memory=False, autocast enabled.")

    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    results = []
    for lam in args.lambdas:
        res = run_experiment(
            lambda_sparse = lam,
            train_loader  = train_loader,
            test_loader   = test_loader,
            device        = device,
            epochs        = args.epochs,
            lr            = args.lr,
            dropout_rate  = args.dropout,
            seed          = args.seed,
            save_dir      = args.out_dir,
        )
        results.append(res)

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Lambda':>12} {'Test Acc':>12} {'Sparsity (%)':>14}")
    print("-" * 42)
    best_model_idx = 0
    for i, res in enumerate(results):
        print(f"{res['lambda']:>12.0e} {res['test_acc']:>11.4%} "
              f"{res['sparsity']:>13.4%}")
        if res["test_acc"] > results[best_model_idx]["test_acc"]:
            best_model_idx = i

    print("\nGenerating plots ...")

    for res in results:
        fname = os.path.join(args.out_dir,
                             f"gate_dist_lambda_{res['lambda']:.0e}.png")
        plot_gate_distribution(res["gate_values"], res["lambda"], fname)

    plot_training_curves(results,
                         os.path.join(args.out_dir, "training_curves.png"))

    csv_path = os.path.join(args.out_dir, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write("lambda,test_accuracy,sparsity_pct\n")
        for res in results:
            f.write(f"{res['lambda']},{res['test_acc']:.6f},"
                    f"{res['sparsity']*100:.4f}\n")
    print(f"  [CSV saved]  {csv_path}")

    print("\nAll outputs written to:", args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()