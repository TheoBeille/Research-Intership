
import torch

from training.loss import trajectory_loss
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=1e-5
                )
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

  
            AxCx, residuals = model(initial_state,functions)


                
            res=residuals[-1]
            weights = torch.linspace(0.1, 1.0, len(AxCx))  # poids croissants
            loss = sum(w * k for w, k in zip(weights, AxCx))



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
        scheduler.step()
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
                    kkt, residuals = model(initial_state,functions)
                    weights = torch.linspace(0.1, 1.0, len(kkt))  # poids croissants
                    loss = sum(w * k for w, k in zip(weights, kkt))
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        val_batches += 1
            val_loss = val_loss / val_batches if val_batches > 0 else float("nan")
            val_loss_hist.append(val_loss)
            model.train()
        
    print("\nTraining finished.")

    return model, train_loss_hist, val_loss_hist