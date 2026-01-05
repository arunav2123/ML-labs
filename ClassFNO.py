import os
import pickle
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.weights1.size(1), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNOBackbone(nn.Module):
    def __init__(self, RR, ZZ, modes=12, width=32, depth=4):
        super().__init__()
        self.width = width

        R_mean = RR.mean()
        Z_mean = ZZ.mean()
        self.register_buffer("RR_centered", torch.from_numpy(RR - R_mean).float())
        self.register_buffer("ZZ_centered", torch.from_numpy(ZZ - Z_mean).float())
        self.lift = nn.Conv2d(3, width, 1) 
        self.fno_blocks = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(depth)])
        self.skips = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])

    def forward(self, x):
        batch_size = x.shape[0]
        
        r = self.RR_centered.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        z = self.ZZ_centered.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        x = torch.cat([x, r, z], dim=1)
        x = self.lift(x)
        for spec, skip in zip(self.fno_blocks, self.skips):
            x = F.gelu(spec(x) + skip(x))
        return x

class ClassFNO(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Linear(backbone.width, 128), Swish(), nn.Linear(128, 1))
        self.boundary_psi = nn.Sequential(nn.Linear(backbone.width, 128), Swish(), nn.Linear(128, 1))

    def forward(self, x):
        features = torch.mean(self.backbone(x), dim=(-2, -1))
        return self.classifier(features), self.boundary_psi(features)

class ClassDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 2)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx, :1], self.y[idx, 1:]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_file = 'ITER_like_equilibrium_dataset_updated.mat'
    checkpoint_path = 'classfno_checkpoint.pth'
    pred_flux_path = 'reconstructed_dataset.mat'
    split_indices_path = 'data_split_indices.pkl'

    print("Loading data...")
    mat = scipy.io.loadmat(data_file)
    RR_pixels = mat['RR_pixels'].astype('float32')
    ZZ_pixels = mat['ZZ_pixels'].astype('float32')

    if os.path.exists(pred_flux_path):
        pred_flux = scipy.io.loadmat(pred_flux_path)
        X_data = pred_flux['all'].squeeze()
    else:
        X_data = mat['DB_psi_pixel_test_ConvNet'].astype('float32')
    y_data = np.column_stack([mat["XP_YN"].T, mat['DB_psi_LCFS_test_ConvNet'].astype('float32')])
    del mat

    if os.path.exists(split_indices_path):
        print("Loading existing split...")
        with open(split_indices_path, 'rb') as f:
            idx = pickle.load(f)
        id_train, id_test = idx['train'], idx['test']
    else:
        print("Creating new split...")
        id_train, id_test = train_test_split(np.arange(len(X_data)), test_size=0.3, random_state=42)
        with open(split_indices_path, 'wb') as f:
            pickle.dump({'train': id_train, 'test': id_test}, f)

    x_mean = X_data[id_train].mean()
    x_std = X_data[id_train].std()
    
    X_data_scaled = (X_data - x_mean) / (x_std)
    print(f"Data scaled: mean={x_mean:.4f}, std={x_std:.4f}")

    train_loader = DataLoader(ClassDataset(X_data_scaled[id_train], y_data[id_train]), batch_size=32, shuffle=True)
    test_loader = DataLoader(ClassDataset(X_data_scaled[id_test], y_data[id_test]), batch_size=32, shuffle=False)

    backbone = FNOBackbone(RR_pixels, ZZ_pixels).to(device)
    model = ClassFNO(backbone).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    loss_funcs = [nn.BCEWithLogitsLoss(), nn.MSELoss()]
    history = {'train_loss': [], 'test_acc': []}

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        history = ckpt.get('history', history)
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch}")

    num_epochs = 100
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for x, t_bin, t_psi in train_loader:
            x, t_bin, t_psi = x.to(device), t_bin.to(device), t_psi.to(device)
            optimizer.zero_grad()
            p_bin, p_psi = model(x)
            loss = loss_funcs[0](p_bin, t_bin) + loss_funcs[1](p_psi, t_psi)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, t_bin, _ in test_loader:
                p_bin, _ = model(x.to(device))
                correct += ((p_bin > 0).float() == t_bin.to(device)).sum().item()
        
        avg_loss = total_loss/len(train_loader)
        acc = correct/len(id_test)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(acc)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Acc {acc:.2%}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history
        }, checkpoint_path)

    def get_predictions(loader):
        p_bins, t_bins, p_psis, t_psis = [], [], [], []
        
        model.eval()
        with torch.no_grad():
            for x, t_bin, t_psi in loader:
                p_bin, p_psi = model(x.to(device))
                p_bins.append(torch.sigmoid(p_bin).cpu().numpy())
                t_bins.append(t_bin.cpu().numpy())
                p_psis.append(p_psi.cpu().numpy())
                t_psis.append(t_psi.cpu().numpy())
        return np.concatenate(p_bins), np.concatenate(t_bins), np.concatenate(p_psis), np.concatenate(t_psis)

    train_p_bin, train_t_bin, train_p_psi, train_t_psi = get_predictions(train_loader)
    val_p_bin, val_t_bin, val_p_psi, val_t_psi = get_predictions(test_loader)

    print("\n--- Equilibrium Classification Performance ---")
    for name, t_bin, p_bin in [("Training", train_t_bin, train_p_bin), ("Validation", val_t_bin, val_p_bin)]:
        print(f"{name} Classification -> Accuracy: {accuracy_score(t_bin, p_bin > 0.5):.4f}, "
              f"F1 Score: {f1_score(t_bin, p_bin > 0.5):.4f}, ROC AUC: {roc_auc_score(t_bin, p_bin):.4f}")

    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.semilogy(history['train_loss'], label='Loss', color='blue')
    plt.title('Training History'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, which="both", ls="-")

    plt.subplot(1, 3, 2)
    plt.plot(history['test_acc'], color='orange', label='Accuracy')
    plt.title('Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.scatter(train_t_psi, train_p_psi, alpha=0.2, s=2, label=f'Train ($R^2$: {r2_score(train_t_psi, train_p_psi):.4f})', color='green')
    plt.scatter(val_t_psi, val_p_psi, alpha=0.5, s=2, label=f'Val ($R^2$: {r2_score(val_t_psi, val_p_psi):.4f})', color='red')
    lims = [min(train_t_psi.min(), val_t_psi.min()), max(train_t_psi.max(), val_t_psi.max())]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    plt.title('Psi_b Reconstruction'); plt.xlabel('Reference'); plt.ylabel('Predicted'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('class_results.png')
    print("\nGraphs saved to class_results.png")
