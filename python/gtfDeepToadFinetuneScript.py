"""
gtfDeepToadFinetuneScript.py

Python/PyTorch port of gtfDeepToadFinetuneScript.m.
Fine-tunes ResNet50 end-to-end on spectrogram images in train_jpg_py.

Split  : 70 / 15 / 15  (train / val / test), stratified per class
Augment: random left-right flip on training images (augmentLR flag)
Optim  : SGD + momentum, piecewise LR decay, early stopping on val loss
Plots  : sample grids, training curves, confusion matrix, ROC, threshold sweep,
         FP / FN image panels

Dependencies
------------
    torch torchvision scikit-learn matplotlib pillow tqdm
    mamba install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
    mamba install scikit-learn matplotlib pillow tqdm
"""

import copy
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc as sk_auc
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
SEED           = 42
DATA_DIR       = Path(r'D:\drbohnen\BetaJun26\train_jpg_py')
OUT_DIR        = Path(__file__).parent
AUGMENT_LR     = True    # random left-right flip on training images
BATCH_SIZE     = 64
MAX_EPOCHS     = 30
LR             = 1e-4
LR_HEAD_FACTOR = 10      # new FC head gets LR * 10 (matches MATLAB WeightLearnRateFactor)
MOMENTUM       = 0.9
L2             = 1e-4
LR_DROP_PERIOD = 5       # epochs between LR drops
LR_DROP_FACTOR = 0.1
VAL_PATIENCE   = 8
FP_WEIGHT      = 1.075
FN_WEIGHT      = 1.0
THR_METHOD     = 'F1 plateau'   # 'max F1' | 'min cost' | 'F1 plateau'
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f'Device : {DEVICE}')
print(f'Data   : {DATA_DIR}')


# ============================================================
# Dataset helpers
# ============================================================

def stratified_split(dataset, fracs, seed=SEED):
    """
    Split an ImageFolder dataset into len(fracs) Subsets,
    stratified by class label.  fracs must sum to ≤ 1.
    """
    rng = random.Random(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    splits = [[] for _ in fracs]
    for indices in class_indices.values():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        boundaries = [0]
        for f in fracs[:-1]:
            boundaries.append(boundaries[-1] + round(f * n))
        boundaries.append(n)
        for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            splits[i].extend(indices[s:e])

    return [Subset(dataset, idx_list) for idx_list in splits]


def get_path(subset, i):
    """Return the file path for the i-th sample in a Subset."""
    global_idx = subset.indices[i]
    path, _ = subset.dataset.samples[global_idx]
    return path


def get_label(subset, i):
    """Return the integer label for the i-th sample in a Subset."""
    global_idx = subset.indices[i]
    _, label = subset.dataset.samples[global_idx]
    return label


# ============================================================
# Transforms
# ============================================================
# ImageNet normalisation (torchvision ResNet50 was trained with these)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip() if AUGMENT_LR else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ============================================================
# Load dataset
# ============================================================
# Use eval_tf as the base (transforms are swapped per split below)
full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=eval_tf)

# Remap labels so bwhistle=1 (positive), other=0 (negative).
# ImageFolder assigns alphabetically (bwhistle=0, other=1) — override that here.
classes = ['other', 'bwhistle']                      # other=0, bwhistle=1
full_dataset.classes     = classes
full_dataset.class_to_idx = {'other': 0, 'bwhistle': 1}
full_dataset.samples     = [(p, full_dataset.class_to_idx[Path(p).parent.name])
                             for p, _ in full_dataset.samples]
full_dataset.targets     = [lbl for _, lbl in full_dataset.samples]

pos_idx   = 1    # bwhistle
other_idx = 0    # other
print(f'\nClasses : {classes}  (pos_idx={pos_idx})')
for cls, cnt in zip(classes, np.bincount([lbl for _, lbl in full_dataset.samples])):
    print(f'  {cls}: {cnt}')

# ---- 70 / 15 / 15 split ----
train_sub, val_sub, test_sub = stratified_split(full_dataset, [0.70, 0.15, 0.15])
print(f'\nSplit  — train: {len(train_sub)}  val: {len(val_sub)}  test: {len(test_sub)}')

# Apply the correct transform to each split by wrapping in a helper dataset
class _TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, i):
        img, label = self.subset[i]
        # img is already a tensor from eval_tf; we need PIL for train_tf
        # Re-read from disk to apply the right transform
        path = get_path(self.subset, i)
        img  = Image.open(path).convert('RGB')
        return self.transform(img), label

train_ds = _TransformSubset(train_sub, train_tf)
val_ds   = _TransformSubset(val_sub,   eval_tf)
test_ds  = _TransformSubset(test_sub,  eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)


# ============================================================
# Plot 1 & 2 — sample image grids (4×4)
# ============================================================
def show_sample_grid(subset, class_name, n=16, seed=SEED):
    rng = random.Random(seed)
    indices = [i for i in range(len(subset))
               if get_label(subset, i) == full_dataset.class_to_idx[class_name]]
    chosen  = rng.sample(indices, min(n, len(indices)))

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(f'{class_name} examples', fontsize=12)
    for ax, i in zip(axes.flat, chosen):
        img = Image.open(get_path(subset, i)).convert('RGB')
        ax.imshow(img); ax.axis('off'); ax.set_title(class_name, fontsize=7)
    for ax in axes.flat[len(chosen):]:
        ax.axis('off')
    plt.tight_layout()

show_sample_grid(train_sub, 'bwhistle')
show_sample_grid(train_sub, 'other')
plt.show(block=False)


# ============================================================
# Build fine-tuned ResNet50
# ============================================================
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Replace 1000-class head with 2-class head
model.fc = nn.Linear(model.fc.in_features, 2)
nn.init.xavier_uniform_(model.fc.weight)
nn.init.zeros_(model.fc.bias)

model = model.to(DEVICE)

# Two parameter groups: backbone at LR, new head at LR * LR_HEAD_FACTOR
backbone_params = [p for n, p in model.named_parameters() if not n.startswith('fc')]
head_params     = list(model.fc.parameters())

optimizer = optim.SGD(
    [{'params': backbone_params, 'lr': LR},
     {'params': head_params,     'lr': LR * LR_HEAD_FACTOR}],
    momentum=MOMENTUM, weight_decay=L2,
)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DROP_PERIOD,
                                        gamma=LR_DROP_FACTOR)
criterion  = nn.CrossEntropyLoss()


# ============================================================
# Training loop
# ============================================================
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss   = float('inf')
patience_count  = 0
best_state      = copy.deepcopy(model.state_dict())
stop_epoch      = MAX_EPOCHS

for epoch in range(1, MAX_EPOCHS + 1):

    # ---- train ----
    model.train()
    run_loss = run_correct = run_total = 0
    for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch:02d} train', leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        run_loss    += loss.item() * imgs.size(0)
        run_correct += (out.argmax(1) == labels).sum().item()
        run_total   += imgs.size(0)

    train_loss = run_loss    / run_total
    train_acc  = run_correct / run_total

    # ---- validate ----
    model.eval()
    run_loss = run_correct = run_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            run_loss    += loss.item() * imgs.size(0)
            run_correct += (out.argmax(1) == labels).sum().item()
            run_total   += imgs.size(0)

    val_loss = run_loss    / run_total
    val_acc  = run_correct / run_total

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f'Epoch {epoch:02d}  '
          f'train loss={train_loss:.4f} acc={train_acc:.3f}  '
          f'val loss={val_loss:.4f} acc={val_acc:.3f}  '
          f'lr={scheduler.get_last_lr()[0]:.2e}')

    # ---- early stopping ----
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        best_state     = copy.deepcopy(model.state_dict())
    else:
        patience_count += 1
        if patience_count >= VAL_PATIENCE:
            stop_epoch = epoch
            print(f'Early stopping at epoch {epoch} (patience={VAL_PATIENCE})')
            break

    scheduler.step()

model.load_state_dict(best_state)
print(f'\nTraining stopped at epoch {stop_epoch}. Best val loss: {best_val_loss:.4f}')


# ============================================================
# Plot 3 — training curves
# ============================================================
epochs_ran = range(1, len(history['train_loss']) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_ran, history['train_loss'], label='Train')
ax1.plot(epochs_ran, history['val_loss'],   label='Val')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-entropy loss')
ax1.set_title('Loss'); ax1.legend(); ax1.grid(True)

ax2.plot(epochs_ran, history['train_acc'], label='Train')
ax2.plot(epochs_ran, history['val_acc'],   label='Val')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True)

fig.suptitle('Fine-tuned ResNet50 — Training progress')
plt.tight_layout()
plt.show(block=False)


# ============================================================
# Evaluate on test set
# ============================================================
model.eval()
all_labels = []
all_preds  = []
all_scores = []   # softmax probabilities

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc='Evaluating test set', leave=False):
        imgs = imgs.to(DEVICE)
        out  = model(imgs)
        prob = torch.softmax(out, dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_scores.extend(prob)

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_scores = np.array(all_scores)      # shape [N, 2]
p          = all_scores[:, pos_idx]    # P(bwhistle) for each test sample

# ---- confusion matrix at default 0.5 threshold ----
cm_raw = confusion_matrix(all_labels, all_preds)
print('\nTest confusion matrix (row-normalised):')
print(cm_raw / cm_raw.sum(axis=1, keepdims=True))

# bwhistle=row0/col0, other=row1/col1
TP = cm_raw[0, 0]; FN_c = cm_raw[0, 1]; FP_c = cm_raw[1, 0]
precision = TP / (TP + FP_c)
recall    = TP / (TP + FN_c)
F1_score  = 2 * precision * recall / (precision + recall)
print(f'Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {F1_score:.3f}')


# ============================================================
# Plot 4 — confusion matrix
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=classes)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
# Add row-normalised values as a second annotation
n_rows = cm_raw.shape[0]
cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)
for i in range(n_rows):
    for j in range(n_rows):
        ax.text(j, i + 0.3, f'({cm_norm[i,j]:.2f})',
                ha='center', va='center', fontsize=8, color='white' if cm_raw[i,j] > cm_raw.max()/2 else 'black')
ax.set_title('Fine-tuned ResNet50 — Test Confusion Matrix')
plt.tight_layout()
plt.show(block=False)


# ============================================================
# ROC — compute now, plot after threshold sweep
# ============================================================
# sklearn roc_curve: pos_label=pos_idx (bwhistle=0)
fpr, tpr, T_roc = roc_curve(all_labels, p, pos_label=pos_idx)
roc_auc = sk_auc(fpr, tpr)
print(f'AUC: {roc_auc:.4f}')


# ============================================================
# Threshold sweep
# ============================================================
thresholds = np.arange(0.05, 0.96, 0.01)
prec_t = np.full_like(thresholds, np.nan)
rec_t  = np.full_like(thresholds, np.nan)
f1_t   = np.full_like(thresholds, np.nan)
cost_t = np.full_like(thresholds, np.nan)
N_test = len(all_labels)

for k, thr in enumerate(thresholds):
    # p >= thr → predict bwhistle (pos_idx); below → predict other (other_idx)
    pl   = np.where(p >= thr, pos_idx, other_idx)
    tp   = int(((pl == pos_idx) & (all_labels == pos_idx)).sum())
    fn_k = int(((pl == other_idx) & (all_labels == pos_idx)).sum())
    fp_k = int(((pl == pos_idx) & (all_labels == other_idx)).sum())
    denom  = tp + fp_k
    if denom > 0:
        prec_t[k] = tp / denom
    rec_t[k]  = tp / (tp + fn_k) if (tp + fn_k) > 0 else 0.0
    if not np.isnan(prec_t[k]) and (prec_t[k] + rec_t[k]) > 0:
        f1_t[k] = 2 * prec_t[k] * rec_t[k] / (prec_t[k] + rec_t[k])
    cost_t[k] = (fp_k * FP_WEIGHT + fn_k * FN_WEIGHT) / N_test

best_cost_idx = int(np.nanargmin(cost_t))
best_f1_idx   = int(np.nanargmax(f1_t))

if THR_METHOD == 'min cost':
    bestThr = thresholds[best_cost_idx]
elif THR_METHOD == 'max F1':
    bestThr = thresholds[best_f1_idx]
elif THR_METHOD == 'F1 plateau':
    plateau_mask = f1_t >= (np.nanmax(f1_t) - 0.001)
    plateau_thrs = thresholds[plateau_mask]
    bestThr      = plateau_thrs[len(plateau_thrs) // 2]
else:
    raise ValueError(f"THR_METHOD must be 'max F1', 'min cost', or 'F1 plateau'; got '{THR_METHOD}'")

print(f'Threshold {thresholds[best_cost_idx]:.2f} (min cost)   : F1={f1_t[best_cost_idx]:.3f}')
print(f'Threshold {thresholds[best_f1_idx]:.2f} (max F1)     : F1={f1_t[best_f1_idx]:.3f}')
print(f'Threshold {bestThr:.2f} (selected: {THR_METHOD})')


# ============================================================
# Plot 5 — ROC with decision boundary marker
# ============================================================
op_idx  = int(np.argmin(np.abs(T_roc - bestThr)))
op_fpr  = fpr[op_idx]
op_tpr  = tpr[op_idx]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, linewidth=1.5, label=f'AUC = {roc_auc:.4f}')
ax.plot(op_fpr, op_tpr, 'ro', markersize=10, markerfacecolor='r')
ax.text(op_fpr + 0.02, op_tpr - 0.03, f'thr={bestThr:.2f}', fontsize=9)
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title(f'ROC — fine-tuned ResNet50 — Test data  (AUC = {roc_auc:.4f})')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.show(block=False)


# ============================================================
# Plot 6 — threshold sweep (dual y-axis, matches MATLAB layout)
# ============================================================
fig, ax_left = plt.subplots(figsize=(9, 5))
ax_right = ax_left.twinx()

ax_left.plot(thresholds, prec_t, linewidth=1.5, label='Precision')
ax_left.plot(thresholds, rec_t,  linewidth=1.5, label='Recall')
ax_left.plot(thresholds, f1_t,   linewidth=1.5, label='F1')
ax_left.set_ylabel('Rate'); ax_left.set_ylim([0, 1])

ax_right.plot(thresholds, cost_t, '--', linewidth=1.5, color='C3', label='Weighted cost')
ax_right.set_ylabel('Weighted cost per sample')

ax_left.axvline(0.50,    linestyle=':',  color='k',  label='default (0.50)')
ax_left.axvline(bestThr, linestyle='--', color='r',  label=f'min cost ({bestThr:.2f})')

lines1, labels1 = ax_left.get_legend_handles_labels()
lines2, labels2 = ax_right.get_legend_handles_labels()
ax_left.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

ax_left.set_xlabel('Threshold')
ax_left.grid(True)
ax_left.set_title(f'Threshold sweep — fine-tuned  (FP={FP_WEIGHT:.3f}  FN={FN_WEIGHT:.3f})')
plt.tight_layout()
plt.show(block=False)


# ============================================================
# FP / FN image panels
# ============================================================
pl_thresh   = np.where(p >= bestThr, pos_idx, other_idx)
is_pred_pos = pl_thresh  == pos_idx
is_true_pos = all_labels == pos_idx

FP_idx = np.where( is_pred_pos & ~is_true_pos)[0]
FN_idx = np.where(~is_pred_pos &  is_true_pos)[0]

FP_sorted = FP_idx[np.argsort(p[FP_idx])[::-1]]   # highest score first
FN_sorted = FN_idx[np.argsort(p[FN_idx])]          # lowest score first


def plot_error_panel(sorted_idx, title, n=16):
    show = sorted_idx[:n]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(f'{title}  (showing {len(show)} of {len(sorted_idx)})', fontsize=11)
    for ax, i in zip(axes.flat, show):
        img = Image.open(get_path(test_sub, int(i))).convert('RGB')
        ax.imshow(img); ax.axis('off')
        ax.set_title(f'p={p[i]:.2f}', fontsize=7)
    for ax in axes.flat[len(show):]:
        ax.axis('off')
    plt.tight_layout()

    print(f'\n--- {title} (showing {len(show)} of {len(sorted_idx)}) ---')
    print(f'  {"#":>3}  {"p(bwhistle)":>12}  filename')
    for rank, i in enumerate(show, 1):
        fname = Path(get_path(test_sub, int(i))).name
        print(f'  {rank:>3}  {p[i]:>12.4f}  {fname}')


plot_error_panel(FP_sorted, 'False Positives')
plot_error_panel(FN_sorted, 'False Negatives')
plt.show(block=False)


# ============================================================
# Save classifier
# ============================================================
save_path = OUT_DIR / 'gtfclassifier_tuned_py.pt'
torch.save({
    'model_state_dict' : best_state,
    'classes'          : classes,
    'pos_idx'          : pos_idx,
    'bestThr'          : float(bestThr),
    'roc_auc'          : float(roc_auc),
    'history'          : history,
}, save_path)
print(f'\nSaved to {save_path}')

plt.show()   # keep all figures open
