import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import numpy as np
from resnet_cifar import ResNet18
from custom_losses import LDAMLoss
from train_utils import run_training_epoch, log_progress, get_eval_preds

def train_expert(train_set, test_loader, cls_counts, subset_targets, epochs, device, 
                 max_m=0.5, beta=0.9999, reweight_epoch=160, use_mixup=False):
    """
    Wersja 'Brute Force' LDAM:
    1. WeightedRandomSampler od 1. epoki (nie czekamy na DRW).
    2. LDAM Loss z s=30 (bardzo ostre granice).
    3. Gradient Clipping (zapobiega wybuchom przy s=30).
    4. Opcjonalne włączenie wag w lossie dopiero w reweight_epoch (domyślnie 160 = nigdy przy 100 epokach).
    """
    print(f"\n[TRAINER EXPERT] Start: LDAM (s=30) + Sampler od startu | Epochs: {epochs}")
    
    cls_num_list = np.array(cls_counts)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / effective_num
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
    weights = 1.0 / torch.tensor(cls_num_list, dtype=torch.float)
    s_weights = [weights[t] for t in subset_targets]
    sampler = WeightedRandomSampler(s_weights, len(s_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=128, sampler=sampler, num_workers=2)
    
    model = ResNet18().to(device)
    criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=max_m, s=30).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    hist, start = [], time.time()
    
    for epoch in range(epochs):
        if epoch == reweight_epoch:
            print(f"  >> [DRW TRIGGER] Activating Per-Class Weights in LDAM Loss")
            criterion.weight = per_cls_weights
        
        loss, acc = run_training_epoch(model, train_loader, criterion, optimizer, device, 
                                       clip_grad=True, use_mixup=use_mixup)
        scheduler.step()
        hist.append(loss)
        
        if (epoch+1) % 10 == 0: 
            log_progress(epoch, epochs, loss, acc, start, optimizer)
            
    t, p = get_eval_preds(model, test_loader, device)
    return t, p, hist