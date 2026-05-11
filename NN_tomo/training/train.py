
import torch
from NN_tomo.training.loss import convergence_loss, tgv_loss

import os

save_path = "./checkpoints"
os.makedirs(save_path, exist_ok=True)

def train(
    model,
    train_data,
    val_data=None,
    n_epochs=200,
    lr=1e-3,
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
    val_loss_hist = []
    
    print(f"Training on {n_epochs} epochs...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(n_epochs):

        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        for initial_state, _, functions in train_data:

            initial_state = initial_state.to(device)

            optimizer.zero_grad()

  
            F_vals, residuals = model(initial_state,functions)

            loss = convergence_loss(residuals)+tgv_loss(F_vals)

            if not torch.isfinite(loss):
                print(f"[Warning] non-finite loss at epoch {epoch}, skipping batch")
                continue

         
            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


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
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch}.pt")
        
        if val_data is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for initial_state, _, functions in val_data:
                    initial_state = initial_state.to(device)
                    _, residuals = model(initial_state,functions)
                    loss = convergence_loss(residuals)
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        val_batches += 1
            val_loss = val_loss / val_batches if val_batches > 0 else float("nan")
            val_loss_hist.append(val_loss)
            model.train()
        
    print("\nTraining finished.")

    return model, train_loss_hist, val_loss_hist