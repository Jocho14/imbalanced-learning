import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import datetime
from resnet_cifar import ResNet18
from train_utils import run_training_epoch, log_progress, get_eval_preds

def train_rotnet(train_set, test_loader, cls_counts, subset_targets, epochs, device):
    """
    Trenuje model w dwóch fazach:
    1. Self-Supervised Learning (Przewidywanie rotacji) - 20 epok
    2. Supervised Fine-Tuning (Klasyfikacja) - reszta epok
    """
    print(f"[TRAINER ROTNET] Start: SSL (Rotation) -> Balanced Fine-Tuning ({epochs} epochs total)")
    
    weights = 1.0 / torch.tensor(cls_counts, dtype=torch.float)
    s_weights = [weights[t] for t in subset_targets]
    sampler = WeightedRandomSampler(s_weights, len(s_weights), replacement=True)
    
    train_loader = DataLoader(train_set, batch_size=128, sampler=sampler, num_workers=2)
    
    ssl_epochs = 20
    print(f">> Phase 1: Rotation Pre-training ({ssl_epochs} epochs)")
    
    model = ResNet18(num_classes=4).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion_ssl = nn.CrossEntropyLoss()
    
    start = time.time()
    
    for epoch in range(ssl_epochs):
        model.train()
        total_loss = 0
        
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            x90 = torch.rot90(inputs, 1, [2, 3])
            x180 = torch.rot90(inputs, 2, [2, 3])
            x270 = torch.rot90(inputs, 3, [2, 3])
            
            x = torch.cat([inputs, x90, x180, x270], dim=0)
            y = torch.cat([
                torch.zeros(batch_size), 
                torch.ones(batch_size), 
                2*torch.ones(batch_size), 
                3*torch.ones(batch_size)
            ], dim=0).long().to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion_ssl(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch+1) % 5 == 0:
             elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
             print(f"  [SSL {epoch+1}/{ssl_epochs}] Loss: {avg_loss:.4f} | Czas: {elapsed}")

    # ==========================
    # PHASE 2: FINE-TUNING
    # ==========================
    ft_epochs = epochs - ssl_epochs
    print(f">> Phase 2: Fine-Tuning ({ft_epochs} epochs)")
    
    if hasattr(model, 'linear'):
        in_features = model.linear.in_features
        model.linear = nn.Linear(in_features, 10).to(device)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 10).to(device)
        
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    hist = []
    start = time.time()
    
    for epoch in range(ft_epochs):
        loss, acc = run_training_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
        scheduler.step()
        hist.append(loss)
        
        current_epoch_total = ssl_epochs + epoch + 1
        
        if (current_epoch_total) % 10 == 0: 
            log_progress(current_epoch_total, epochs, loss, acc, start, optimizer)
            
    t, p = get_eval_preds(model, test_loader, device)
    return t, p, hist