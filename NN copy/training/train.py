
import torch
from training.loss import convergence_loss, tgv_loss

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
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10  
    )

    train_loss_hist = []
    val_loss_hist = []
    
    print(f"Training on {n_epochs} epochs...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        for batch_idx, (initial_state, _, functions) in enumerate(train_data):
            initial_state = initial_state.to(device)
            optimizer.zero_grad()

            try:
                AxCx, residuals = model(initial_state, functions)
                
 
                if len(AxCx) == 0:
                    print(f"Warning: Empty AxCx at epoch {epoch}, batch {batch_idx}")
                    continue

                AxCx_tensor = torch.stack([
                    torch.as_tensor(a, device=device, dtype=torch.float32) 
                    for a in AxCx
                ])
                
                loss_kkt = torch.mean(AxCx_tensor)
    
                loss = loss_kkt


                if not torch.isfinite(loss):
                    print(f"[Warning] non-finite loss at epoch {epoch}, batch {batch_idx}")
                    print(f" loss_kkt={loss_kkt:.3f}")
                    continue

                loss.backward()
                
    
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            print(f"Warning: non-finite gradient in {name}")
                            p.grad = torch.nan_to_num(p.grad)

                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                
            except RuntimeError as e:
                print(f"RuntimeError at epoch {epoch}, batch {batch_idx}: {e}")
                continue

        # Average loss
        if valid_batches > 0:
            epoch_loss /= valid_batches
        else:
            epoch_loss = float("nan")

        train_loss_hist.append(epoch_loss)


        if val_data is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for initial_state, _, functions in val_data:
                    initial_state = initial_state.to(device)
                    AxCx_val, residuals = model(initial_state, functions)  
                    loss = convergence_loss(residuals)
                    if torch.isfinite(loss):
                        val_loss += loss.item()
                        val_batches += 1
            
            val_loss = val_loss / val_batches if val_batches > 0 else float("nan")
            val_loss_hist.append(val_loss)
            

            scheduler.step(val_loss)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch:4d} | Train={epoch_loss:.6f} | Val={val_loss:.6f}")
        elif epoch % print_every == 0:
            print(f"Epoch {epoch:4d} | Train loss = {epoch_loss:.6f}")
            
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
            }, f"{save_path}/model_epoch_{epoch}.pt")
    
    print("\nTraining finished.")
    return model, train_loss_hist, val_loss_hist