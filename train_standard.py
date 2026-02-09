import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from resnet_cifar import ResNet18
from train_utils import run_training_epoch, log_progress, get_eval_preds

def train_standard(train_loader, test_loader, epochs, device, use_weights=False, cls_counts=None, custom_criterion=None):
    mode = "CUSTOM_LOSS" if custom_criterion else ("WEIGHTED" if use_weights else "BASELINE")
    print(f"\n[TRAINER STANDARD] Start: {mode} ({epochs} epok)")
    
    model = ResNet18().to(device)
    
    if custom_criterion is not None:
        criterion = custom_criterion
    elif use_weights:
        weights = 1.0 / torch.tensor(cls_counts, dtype=torch.float)
        weights = weights / weights.sum() * len(cls_counts)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    hist, start = [], time.time()
    for epoch in range(epochs):
        loss, acc = run_training_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        hist.append(loss)
        
        if (epoch+1) % 10 == 0: 
            log_progress(epoch, epochs, loss, acc, start, optimizer)
            
    targets, preds = get_eval_preds(model, test_loader, device)
            
    return targets, preds, hist