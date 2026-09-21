import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score

def train_and_evaluate(models_dict, extractor_endo, extractor_histo, train_loader, val_loader, device, epochs=20):
    history = {name: {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_recall': [], 'probs': [], 'labels': [], 'preds': []} for name in models_dict}
    
    for name, model in models_dict.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        has_params = len(list(model.parameters())) > 0
        if has_params:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
        criterion_ce = nn.CrossEntropyLoss()
        criterion_nll = nn.NLLLoss()
        
        for epoch in range(epochs):
            model.train() if has_params else model.eval()
            train_loss = 0.0
            
            for endo, histo, labels in train_loader:
                endo, histo, labels = endo.to(device), histo.to(device), labels.to(device)
                if has_params: optimizer.zero_grad()
                
                if "SOTA 1" in name:
                    logits, _ = model(endo, histo)
                    loss = criterion_ce(logits, labels)
                else:
                    with torch.no_grad():
                        f_endo, l_endo = extractor_endo(endo)
                        f_histo, l_histo = extractor_histo(histo)
                    if "SOTA 2" in name: probs, _ = model(l_endo, l_histo)
                    elif "SOTA 3" in name: probs, _ = model(f_endo, f_histo)
                    else: probs, _, _, _ = model(f_endo, f_histo)
                    loss = criterion_nll(torch.log(probs + 1e-10), labels)
                
                if has_params:
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
                
            history[name]['train_loss'].append(train_loss / len(train_loader))
            
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            all_probs_ep, all_labels_ep, all_preds_ep = [], [], []
            
            with torch.no_grad():
                for endo, histo, labels in val_loader:
                    endo, histo, labels = endo.to(device), histo.to(device), labels.to(device)
                    
                    if "SOTA 1" in name:
                        logits, _ = model(endo, histo)
                        loss = criterion_ce(logits, labels)
                        probs_batch = F.softmax(logits, dim=1)
                    else:
                        f_endo, l_endo = extractor_endo(endo)
                        f_histo, l_histo = extractor_histo(histo)
                        if "SOTA 2" in name: probs_batch, _ = model(l_endo, l_histo)
                        elif "SOTA 3" in name: probs_batch, _ = model(f_endo, f_histo)
                        else: probs_batch, _, _, _ = model(f_endo, f_histo)
                        loss = criterion_nll(torch.log(probs_batch + 1e-10), labels)
                    
                    val_loss += loss.item()
                    preds = torch.argmax(probs_batch, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    all_probs_ep.extend(probs_batch[:, 1].cpu().numpy())
                    all_labels_ep.extend(labels.cpu().numpy())
                    all_preds_ep.extend(preds.cpu().numpy())
                        
            epoch_f1 = f1_score(all_labels_ep, all_preds_ep, zero_division=0)
            epoch_recall = recall_score(all_labels_ep, all_preds_ep, zero_division=0)
            epoch_acc = correct / total
            
            history[name]['val_loss'].append(val_loss / len(val_loader))
            history[name]['val_acc'].append(epoch_acc)
            history[name]['val_f1'].append(epoch_f1)
            history[name]['val_recall'].append(epoch_recall)
            
            if epoch == epochs - 1:
                history[name]['probs'] = all_probs_ep
                history[name]['labels'] = all_labels_ep
                history[name]['preds'] = all_preds_ep

        train_time = (time.time() - start_time) / 60
        print(f"{name} | Val Acc: {history[name]['val_acc'][-1]*100:.2f}% | Time: {train_time:.1f}m\n")

    return history