import torch
import os
import argparse
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import numpy as np

# ----------------------------------------
# Configuration
# ----------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------------------
# Helper Dataset
# ----------------------------------------
class ImageCSV(Dataset):
    def __init__(self, csv_path, root_dir="", transform=None):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row.iloc[0] 
        label = row.iloc[1]
        path = os.path.join(self.root, fname) if self.root else fname
        
        try:
            image = Image.open(path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, label, str(path)

# ----------------------------------------
# Model Loader
# ----------------------------------------
def build_model(num_classes=2, device="cpu", weights_load_path=None):
    model = models.resnet50(weights=None)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    if weights_load_path:
        state = torch.load(weights_load_path, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Loaded weights: {weights_load_path}")
        
    model.to(device)
    model.eval()
    return model

# ----------------------------------------
# Evaluation (TTA + Auto-Threshold)
# ----------------------------------------
def evaluate(model, dataloader, device, class_names, save_csv_path=None):
    print(f"\n--- Evaluation Started (TTA Enabled) ---")
    
    all_true = []
    all_probs = []
    all_paths = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0: print(f"Processing batch {i}...", end='\r')
            
            if len(batch) == 3: imgs, labels, paths = batch
            else: imgs, labels = batch; paths = [""] * len(labels)

            imgs = imgs.to(device)

            # --- TTA: FLIP & AVERAGE ---
            out1 = model(imgs)
            prob1 = softmax(out1)

            imgs_flip = torch.flip(imgs, [3])
            out2 = model(imgs_flip)
            prob2 = softmax(out2)

            avg_prob = (prob1 + prob2) / 2
            
            tb_probs = avg_prob[:, 1].cpu().numpy()
            
            for k in range(len(labels)):
                lab = labels[k].item() if isinstance(labels[k], torch.Tensor) else labels[k]
                all_true.append(lab)
                all_probs.append(float(tb_probs[k]))
                all_paths.append(paths[k])

    y_true = np.array(all_true)
    y_scores = np.array(all_probs)

    # --- AUTO-THRESHOLD ---
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    
    print(f"\n\n🔥 Auto-Threshold Detected: {best_thresh:.4f}")
    
    preds = (y_scores >= best_thresh).astype(int)

    # Metrics
    acc = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, target_names=class_names, zero_division=0)
    
    print(f"\n{'='*20} RESULTS {'='*20}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    
    tn, fp, fn, tp = cm.ravel()
    norm_prec = tn / (tn + fn) if (tn+fn) > 0 else 0
    print(f"\nNormal Precision: {norm_prec:.4f}")

    if save_csv_path:
        df = pd.DataFrame({
            "path": all_paths,
            "true_label": [class_names[i] for i in y_true],
            "tb_prob": y_scores,
            "pred_label": [class_names[i] for i in preds]
        })
        df.to_csv(save_csv_path, index=False)
        print(f"Predictions saved to {save_csv_path}")

# ----------------------------------------
# Main
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-csv", type=str, default="predictions_tta.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class-names", type=str, nargs="+", default=["Normal", "Tuberculosis"])
    args = parser.parse_args()

    # --- MATCHING PREPROCESSING ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    if args.data_dir:
        dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    elif args.csv:
        dataset = ImageCSV(args.csv, transform=transform)
    else:
        raise ValueError("Provide --data-dir or --csv")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = build_model(num_classes=len(args.class_names), device=args.device, weights_load_path=args.weights)
    
    evaluate(model, dataloader, args.device, args.class_names, save_csv_path=args.save_csv)

if __name__ == "__main__":
    main()