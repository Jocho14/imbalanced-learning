import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
from resnet_cifar import ResNet18
from train_utils import run_training_epoch, log_progress, get_eval_preds

def train_pro(train_set, test_loader, cls_counts, subset_targets, epochs, device, custom_criterion=None):
    print(f"\n[TRAINER PRO] Start: MixUp + Sampler ({epochs} epok)")
    
    weights = 1.0 / torch.tensor(cls_counts, dtype=torch.float)
    s_weights = [weights[t] for t in subset_targets]
    sampler = WeightedRandomSampler(s_weights, len(s_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=128, sampler=sampler, num_workers=2)
    
    model = ResNet18().to(device)
    
    if custom_criterion is not None:
        criterion = custom_criterion
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    hist, start = [], time.time()
    
    for epoch in range(epochs):
        loss, acc = run_training_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
        scheduler.step()
        hist.append(loss)
        
        if (epoch+1) % 10 == 0:
            log_progress(epoch, epochs, loss, acc, start, optimizer)

    targets, preds = get_eval_preds(model, test_loader, device)
            
    return targets, preds, hist