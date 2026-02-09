import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
from resnet_cifar import ResNet18
from train_utils import get_eval_preds
from custom_losses import LDAMLoss

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        return out

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_hybrid(train_set, test_loader, cls_counts, subset_targets, epochs, device):
    epochs = 160
    drw_epoch = 120 
    
    print(f"\n[TRAINER V4 STABLE] Start: DRW (Switch @ {drw_epoch}) + LDAM (s=10) + MixUp (Always ON)")
    
    model = ResNet18().to(device)
    num_classes = len(cls_counts)
    
    if hasattr(model, 'linear'):
        in_features = model.linear.in_features
        model.linear = NormedLinear(in_features, num_classes).to(device)
    else:
        in_features = model.fc.in_features
        model.fc = NormedLinear(in_features, num_classes).to(device)

    criterion = LDAMLoss(cls_num_list=cls_counts, max_m=0.5, s=10.0).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 150], gamma=0.1)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    
    weights_np = 1.0 / np.array(cls_counts)
    sample_weights = weights_np[subset_targets]
    balanced_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    start = time.time()
    using_balanced = False
    
    hist = [] 

    for epoch in range(1, epochs + 1):
        if epoch == drw_epoch and not using_balanced:
            print(f"  [DRW TRIGGER] Epoch {epoch}: Switching to WeightedRandomSampler & Decaying LR")
            train_loader = DataLoader(train_set, batch_size=128, sampler=balanced_sampler, num_workers=2)
            using_balanced = True
        
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # MixUp
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        hist.append(epoch_loss) 
        
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch >= 140 or epoch == drw_epoch:
             model.eval()
             correct_val = 0
             total_val = 0
             with torch.no_grad():
                 for v_inputs, v_targets in test_loader:
                     v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                     v_outputs = model(v_inputs)
                     _, v_predicted = v_outputs.max(1)
                     total_val += v_targets.size(0)
                     correct_val += v_predicted.eq(v_targets).sum().item()
             
             val_acc = correct_val / total_val
             
             elapsed = time.time() - start
             eta = elapsed / epoch * (epochs - epoch)
             m, s = divmod(int(eta), 60)
             
             print(f"  [Epoka {epoch}/{epochs}] Loss: {epoch_loss:.4f} | Acc: {val_acc*100:.2f}% | LR: {current_lr:.5f} | ETA: {int(m)}m {int(s)}s")

    t, p = get_eval_preds(model, test_loader, device)
    
    return t, p, hist