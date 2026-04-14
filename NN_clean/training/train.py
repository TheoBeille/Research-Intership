# training/train.py

import torch
from training.loss import convergence_loss


def train(
    model,
    train_data,
    n_epochs=100,
    lr=1e-4,
    device="cuda",
    grad_clip=1.0,
    print_every=10,
):
    """
    Train the unrolled model.

    Args:
        model: UnrolledFBS
        train_data: list of (noisy, clean, functions)
        n_epochs: number of epochs
        lr: learning rate
        device: cpu/cuda
    """

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    train_loss_hist = []

    print(f"Training on {n_epochs} epochs...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(n_epochs):

        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        for noisy, _, functions in train_data:

            noisy = noisy.to(device)

            optimizer.zero_grad()

            # forward pass
            _, residuals = model(noisy, functions)

            loss = convergence_loss(residuals)

            if not torch.isfinite(loss):
                print(f"[Warning] non-finite loss at epoch {epoch}, skipping batch")
                continue

            # backward
            loss.backward()

            # gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # safe gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad)

            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

        # average loss
        if valid_batches > 0:
            epoch_loss /= valid_batches
        else:
            epoch_loss = float("nan")

        train_loss_hist.append(epoch_loss)

        if epoch % print_every == 0:
            print(f"Epoch {epoch:4d} | Train loss = {epoch_loss:.6f}")

    print("\nTraining finished.")

    return model, train_loss_hist