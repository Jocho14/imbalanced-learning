import torch
import torch.nn as nn
import time
import datetime
import numpy as np

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_eval_preds(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return np.array(all_targets), np.array(all_preds)

def log_progress(epoch, epochs, loss, acc, start_time, optimizer=None):
    elapsed = time.time() - start_time
    avg_time = elapsed / (epoch + 1)
    remaining = avg_time * (epochs - epoch - 1)
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    rem_str = str(datetime.timedelta(seconds=int(remaining)))
    
    lr_info = ""
    if optimizer:
        lr = optimizer.param_groups[0]['lr']
        lr_info = f"| LR: {lr:.5f} "
        
    print(f"  [Epoka {epoch+1}/{epochs}] Loss: {loss:.4f} | Acc: {acc:.1f}% {lr_info}| Czas: {elapsed_str} (ETA: {rem_str})")

def run_training_epoch(model, loader, criterion, optimizer, device, use_mixup=False, clip_grad=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if use_mixup:
            inputs, ta, tb, lam = mixup_data(inputs, targets, device=device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, ta, tb, lam)
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(ta).float() + (1 - lam) * predicted.eq(tb).float()).sum().item()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        loss.backward()
        if clip_grad: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total += targets.size(0)
    return total_loss / len(loader), 100. * correct / total