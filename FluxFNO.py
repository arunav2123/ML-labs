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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
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

class MixedFluxFNO(nn.Module):
    def __init__(self, backbone, num_scalars=100):
        super().__init__()
        self.backbone = backbone
        self.scalar_encoder = nn.Sequential(
            nn.Linear(num_scalars, 128), Swish(),
            nn.Linear(128, backbone.width), Swish()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(backbone.width * 2, 64, 1), Swish(),
            nn.Conv2d(64, 32, 1), Swish(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, pressure, scalars):
        scalar_feat = self.scalar_encoder(scalars)
        x = self.backbone(pressure)
        b, c, h, w = x.shape
        scalar_map = scalar_feat.view(b, c, 1, 1).expand(b, c, h, w)
        fused = torch.cat((x, scalar_map), dim=1)
        return self.decoder(fused)

class MixedFluxDataset(Dataset):
    def __init__(self, X_pressure, X_scalars, y):
        self.X_pressure = torch.tensor(X_pressure, dtype=torch.float32).unsqueeze(1)
        self.X_scalars = torch.tensor(X_scalars, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X_pressure)
    def __getitem__(self, idx):
        return self.X_pressure[idx], self.X_scalars[idx], self.y[idx]

def gs_kernels(dr, dz, device):
    alpha = -2.0 * (dr**2 + dz**2)
    beta = (dr**2 * dz**2) / alpha

    laplace_kernel = torch.tensor([
        [0.0, dr**2/alpha, 0.0],
        [dz**2/alpha, 1.0, dz**2/alpha],
        [0.0, dr**2/alpha, 0.0]
    ], device=device).view(1, 1, 3, 3)

    scale_dr = (dr**2 * dz**2) / (2.0 * alpha * dr)
    dr_kernel = torch.tensor([[0.0, 0.0, 0.0],[1.0, 0.0, -1.0],[0.0, 0.0, 0.0]], device=device).view(1, 1, 3, 3) * scale_dr
    return laplace_kernel, dr_kernel, beta

class CombinedLoss(nn.Module):
    def __init__(self, RR_pixels, dr, dz, device="cuda"):
        super().__init__()
        self.register_buffer("R", torch.from_numpy(RR_pixels).to(device).view(1, 1, 64, 64))
        self.k_lap, self.k_dr, self.beta = gs_kernels(dr, dz, device)

    def compute_gs_op(self, psi):
        term_lap = F.conv2d(psi, self.k_lap, padding=1)
        term_dr = (1.0 / (self.R + 1e-6)) * F.conv2d(psi, self.k_dr, padding=1)
        return (1.0 / self.beta) * (term_lap - term_dr)

    def forward(self, psi_pred, psi_true, w_gs=0.1):
        loss_psi = F.mse_loss(psi_pred, psi_true)
        gs_pred = self.compute_gs_op(psi_pred)
        gs_true = self.compute_gs_op(psi_true)
        loss_gs = F.mse_loss(gs_pred, gs_true)
        return loss_psi + w_gs * loss_gs, loss_psi, loss_gs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_file = 'ITER_like_equilibrium_dataset_updated.mat'
    checkpoint_path = 'fluxfno_checkpoint.pth'
    split_indices_path = 'data_split_indices.pkl'

    print("Loading data...")
    mat = scipy.io.loadmat(data_file)

    RR_pixels = mat['RR_pixels'].astype('float32')
    ZZ_pixels = mat['ZZ_pixels'].astype('float32')
    dr = (RR_pixels[0, 1] - RR_pixels[0, 0]).item()
    dz = (ZZ_pixels[1, 0] - ZZ_pixels[0, 0]).item()

    Xp_data = mat['DB_p_2D_test_ConvNet'].astype('float32')
    Xs_data = np.column_stack([mat['DB_meas_Bpickup_test_ConvNet'], mat['DB_coils_curr_test_ConvNet']]).astype('float32')
    y_data = mat['DB_psi_pixel_test_ConvNet'].astype('float32')
    del mat

    if os.path.exists(split_indices_path):
        print("Loading existing split...")
        with open(split_indices_path, 'rb') as f:
            idx = pickle.load(f)
        id_train, id_test = idx['train'], idx['test']
    else:
        print("Creating new split...")
        id_train, id_test = train_test_split(np.arange(len(Xs_data)), test_size=0.3, random_state=42)
        with open(split_indices_path, 'wb') as f:
            pickle.dump({'train': id_train, 'test': id_test}, f)

    n_r = y_data.shape[1]
    n_z = y_data.shape[2]

    x_mean = Xp_data[id_train].mean()
    x_std = Xp_data[id_train].std()

    Xp_data_scaled = (Xp_data - x_mean) / (x_std)
    print(f"Data scaled: mean={x_mean:.4f}, std={x_std:.4f}")

    scaler = StandardScaler()
    scaler.fit(Xs_data[id_train])
    Xs_data_scaled = scaler.transform(Xs_data)

    train_loader = DataLoader(MixedFluxDataset(Xp_data_scaled[id_train], Xs_data_scaled[id_train], y_data[id_train]), batch_size=32, shuffle=True)
    test_loader = DataLoader(MixedFluxDataset(Xp_data_scaled[id_test], Xs_data_scaled[id_test], y_data[id_test]), batch_size=32, shuffle=False)

    backbone = FNOBackbone(RR_pixels, ZZ_pixels).to(device)
    model = MixedFluxFNO(backbone, num_scalars=Xs_data.shape[1]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    loss_func = CombinedLoss(RR_pixels, dr, dz, device=device)
    history = {'train_loss': [], 'val_loss': []}

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        history = ckpt.get('history', history)
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch}")

    num_epochs = 300
    w_gs = 0.1
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_stats = {'loss': 0.0, 'psi': 0.0, 'gs': 0.0}

        for p_2d, scalars, y_true in train_loader:
            p_2d, scalars, y_true = p_2d.to(device), scalars.to(device), y_true.to(device)
            optimizer.zero_grad()
            psi_pred = model(p_2d, scalars)

            loss, l_psi, l_gs = loss_func(psi_pred, y_true, w_gs=w_gs)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_stats['loss'] += loss.item()
            train_stats['psi'] += l_psi.item()
            train_stats['gs'] += l_gs.item()

        scheduler.step()

        model.eval()
        val_stats = {'loss': 0.0, 'psi': 0.0, 'gs': 0.0}
        with torch.no_grad():
            for p_2d, scalars, y_true in test_loader:
                p_2d, scalars, y_true = p_2d.to(device), scalars.to(device), y_true.to(device)
                psi_pred = model(p_2d, scalars)
                loss, l_psi, l_gs = loss_func(psi_pred, y_true, w_gs=w_gs)

                val_stats['loss'] += loss.item()
                val_stats['psi'] += l_psi.item()
                val_stats['gs'] += l_gs.item()

        train_loss = train_stats['loss'] / len(train_loader)
        val_loss = val_stats['loss'] / len(test_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.5f} - (Psi: {train_stats['psi']/len(train_loader):.6f}/GS: {train_stats['gs']/len(train_loader):.6f})")
        print(f"--> Test Accuracy: {val_loss:.5f} - (Psi: {val_stats['psi'] / len(test_loader):.6f}/GS: {val_stats['gs'] / len(test_loader):.6f})")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
        }, checkpoint_path)

    def get_metrics(loader):
        p_psis, t_psis, p_gss, t_gss = [], [], [], []

        model.eval()
        with torch.no_grad():
            for p_2d, scalars, y_true in loader:
                p_2d, scalars, y_true = p_2d.to(device), scalars.to(device), y_true.to(device)
                psi_pred = model(p_2d, scalars)
                gs_pred = loss_func.compute_gs_op(psi_pred)
                gs_true = loss_func.compute_gs_op(y_true)
                p_psis.append(psi_pred.cpu().numpy())
                t_psis.append(y_true.cpu().numpy())
                p_gss.append(gs_pred.cpu().numpy())
                t_gss.append(gs_true.cpu().numpy())
        p_psi = np.concatenate(p_psis).flatten()
        t_psi = np.concatenate(t_psis).flatten()
        p_gs = np.concatenate(p_gss).flatten()
        t_gs = np.concatenate(t_gss).flatten()

        results = {
            "Psi_MSE": mean_squared_error(t_psi, p_psi),
            "Psi_R2": r2_score(t_psi, p_psi),
            "GS_MSE": mean_squared_error(t_gs, p_gs),
            "GS_R2": r2_score(t_gs, p_gs),
            "p_psi": p_psi,
            "t_psi": t_psi,
            "p_gs": p_gs,
            "t_gs": t_gs
        }
        return results

    train_metrics = get_metrics(train_loader)
    val_metrics = get_metrics(test_loader)

    print("\n--- Flux Training Benchmarks ---")
    print(f"Psi -> MSE: {train_metrics['Psi_MSE']:.6f}, R2: {train_metrics['Psi_R2']:.6f}")
    print(f"GS -> MSE: {train_metrics['GS_MSE']:.6f}, R2: {train_metrics['GS_R2']:.6f}")

    print("\n--- Flux Validation Benchmarks ---")
    print(f"Psi -> MSE: {val_metrics['Psi_MSE']:.6f}, R2: {val_metrics['Psi_R2']:.6f}")
    print(f"GS -> MSE: {val_metrics['GS_MSE']:.6f}, R2: {val_metrics['GS_R2']:.6f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.semilogy(history['train_loss'], label='Train Loss')
    plt.semilogy(history['val_loss'], label='Val Loss')
    plt.title('Log Training Loss (Total)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(loc='upper right'); plt.grid(True, which="both", ls="-")

    plt.subplot(1, 3, 2)
    plt.scatter(train_metrics['t_gs'], train_metrics['p_gs'], alpha=0.2, s=2, label=f'Train ($R^2$: {train_metrics["GS_R2"]:.4f})', color='green')
    plt.scatter(val_metrics['t_gs'], val_metrics['p_gs'], alpha=0.2, s=2, label=f'Val ($R^2$: {train_metrics["GS_R2"]:.4f})', color='red')
    lims = [min(train_metrics['t_gs'].min(), val_metrics['t_gs'].min()), max(train_metrics['t_gs'].max(), val_metrics['t_gs'].max())]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    plt.title('Grad-Shafranov'); plt.xlabel('Reference \u0394*\u03A8'); plt.ylabel('Predicted \u0394*\u03A8'); plt.legend(loc='upper left'); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('flux_results.png')
    print("Saved results to flux_results.png")

    final_loader = DataLoader(MixedFluxDataset(Xp_data_scaled, Xs_data_scaled, y_data), batch_size=1024)
    model.eval()
    final_psi = []
    with torch.no_grad():
        for p_2d, scalars, _ in final_loader:
            p_2d, scalars = p_2d.to(device), scalars.to(device)
            final_psi.append(model(p_2d, scalars).cpu().numpy())
    result_path = 'reconstructed_dataset.mat'
    scipy.io.savemat(result_path, {'all':np.concatenate(final_psi)})
    print(f"Saved reconstructed flux to {result_path}")