
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
import os
os.environ["OPENBLAS_NUM_THREADS"] = "24"

# Define VAE-style encoder (without mean/variance, only residual connections)
class VAE_Encoder(nn.Module):
    def __init__(self, num_features, latent_dim, hidden_dim=[512, 512], residual=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(
                        nn.Linear(num_features, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # Output layer, directly generating low-dimensional latent embedding
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x  # residual connection
            else:
                x = layer(x)
        return x  # return low-dimensional embedding


class RNADecoderWithZINB(nn.Module):
    """
    RNA decoder: outputs reconstructed continuous matrix and ZINB parameters
    """

    def __init__(self, num_features, latent_dim, hidden_dim=[512, 512], residual=True):
        super().__init__()
        self.network = nn.ModuleList()
        self.residual = residual

        # Encoding layers
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )

        # Three output heads
        # 1. Reconstructed continuous values
        self.continuous_output = nn.Linear(hidden_dim[-1], num_features)

        # 2. Mean parameter of Negative Binomial distribution (dispersion)
        self.mean_output = nn.Sequential(
            nn.Linear(hidden_dim[-1], num_features),
            nn.Softplus()  # ensure mean is non‑negative
        )

        # 3. Zero‑inflation probability
        self.pi_output = nn.Sequential(
            nn.Linear(hidden_dim[-1], num_features),
            nn.Sigmoid()  # ensure values in [0,1]
        )

    def forward(self, x):
        """Returns: reconstructed continuous values, mean parameter, zero‑inflation probability"""
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)

        continuous_recon = self.continuous_output(x)
        mean_param = self.mean_output(x)
        pi_param = self.pi_output(x)

        return continuous_recon, mean_param, pi_param


class VAE_Decoder_Bernoulli(nn.Module):
    def __init__(self, num_features, latent_dim, hidden_dim=[512, 512], residual=True):
        super().__init__()
        self.network = nn.ModuleList()
        self.residual = residual
        for i in range(len(hidden_dim)):
            if i == 0:
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:
                self.network.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # Output layer, using Sigmoid to output Bernoulli probabilities
        self.network.append(nn.Linear(hidden_dim[-1], num_features))

    def forward(self, x):
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x  # residual connection
            else:
                x = layer(x)
        # Use Sigmoid to output Bernoulli probabilities
        return torch.sigmoid(x)


# GCN graph convolution layer (same as before)
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x, adj):
        support = self.linear(x)  # linear transformation (H * W)
        output = torch.mm(adj, support)  # multiply with adjacency matrix (Ã * (H * W))
        if self.activation is not None:
            output = self.activation(output)
        return output


# Multi‑head attention fusion module
class AttentionFusion(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        # Keep linear layer for generating adjacency matrix
        self.adjacency_layer = nn.Linear(input_dim, input_dim)

    def forward(self, rna_feat, atac_feat):
        # rna_feat, atac_feat: (batch, dim)
        combined = torch.stack([rna_feat, atac_feat], dim=1)   # (batch, 2, dim)
        fused_seq, _ = self.attn(combined, combined, combined) # self‑attention output (batch, 2, dim)
        fused = fused_seq.mean(dim=1)   # average fusion -> (batch, dim)

        # Compute A_fused (similar to original code: linear transform + sigmoid on fused)
        A_fused = torch.sigmoid(self.adjacency_layer(fused))   # (batch, dim)

        return fused, A_fused   # return both values


# Combined model (input RNA and ATAC data)
class MultiModalAutoencoderWithZINB(nn.Module):
    """Improved multi‑modal autoencoder using ZINB loss"""

    def __init__(self, num_rna_features, num_atac_features, vae_latent_dim=256, gcn_latent_dim=128, rna_cont_weight=0.5, zinb_weight=0.3, atac_weight=0.4, num_classes=None):
        super().__init__()

        total_features = num_rna_features + num_atac_features
        # Store loss weights
        self.rna_cont_weight = rna_cont_weight
        self.zinb_weight = zinb_weight
        self.atac_weight = atac_weight
        # VAE encoder and decoders
        self.encoder = VAE_Encoder(total_features, vae_latent_dim)
        self.rna_decoder = RNADecoderWithZINB(num_rna_features, gcn_latent_dim)  # new decoder
        self.atac_decoder = VAE_Decoder_Bernoulli(num_atac_features, gcn_latent_dim)

        # GCN graph convolution layers
        self.gcn_rna = GCNLayer(vae_latent_dim, gcn_latent_dim)
        self.gcn_atac = GCNLayer(vae_latent_dim, gcn_latent_dim)

        # Attention fusion module
        self.attention_fusion = AttentionFusion(gcn_latent_dim)

        # ZINB loss module
        self.zinb_loss = ZINBLossForContinuous(zero_threshold=0.01)

        self.num_classes = num_classes


    def forward(self, rna_expression, atac_expression, rna_adj, atac_adj):
        # Concatenate inputs
        combined_expression = torch.cat((rna_expression, atac_expression), dim=1)

        # Encode
        combined_emb = self.encoder(combined_expression)

        # GCN processing
        rna_gcn_out = self.gcn_rna(combined_emb, rna_adj)
        atac_gcn_out = self.gcn_atac(combined_emb, atac_adj)

        # Attention fusion
        fused_embedding, A_fused = self.attention_fusion(rna_gcn_out, atac_gcn_out)

        # Decode
        # RNA: reconstructed continuous values + ZINB parameters
        rna_continuous_recon, rna_mean, rna_pi = self.rna_decoder(fused_embedding)

        # ATAC: Bernoulli reconstruction
        atac_reconstructed = self.atac_decoder(fused_embedding)


        # Compute losses
        losses = self.compute_loss(
            rna_continuous_recon, rna_mean, rna_pi,
            atac_reconstructed,
            rna_expression, atac_expression,
            A_fused, rna_adj, atac_adj
        )

        return rna_continuous_recon, rna_mean, rna_pi, atac_reconstructed, losses, fused_embedding

    def compute_loss(self, rna_continuous_recon, rna_mean, rna_pi,
                     atac_reconstructed,
                     rna_expression, atac_expression,
                     A_fused, rna_adj, atac_adj):
        # 1. RNA continuous reconstruction loss
        rna_continuous_loss = F.mse_loss(rna_continuous_recon, rna_expression)
        # Check if rna_continuous_loss is NaN or inf
        if torch.isnan(rna_continuous_loss) or torch.isinf(rna_continuous_loss):
            print("rna_continuous_loss is nan or inf")
            print("rna_continuous_recon min/max:", rna_continuous_recon.min().item(), rna_continuous_recon.max().item())
            print("rna_expression min/max:", rna_expression.min().item(), rna_expression.max().item())

        # 2. RNA ZINB loss (based on binarization)
        zinb_loss, binary_matrix, imputed_matrix = self.zinb_loss(
            rna_expression, rna_mean, rna_pi, theta=10.0
        )
        if torch.isnan(zinb_loss) or torch.isinf(zinb_loss):
            print("zinb_loss is nan or inf")
            print("rna_mean min/max:", rna_mean.min().item(), rna_mean.max().item())
            print("rna_pi min/max:", rna_pi.min().item(), rna_pi.max().item())

        # 3. ATAC Bernoulli reconstruction loss
        # Clamp decoder output to avoid log(0) or log(1) causing NaN
        atac_reconstructed_clamped = torch.clamp(atac_reconstructed, 1e-8, 1 - 1e-8)
        atac_expression_clamped = torch.clamp(atac_expression, 1e-8, 1 - 1e-8)
        atac_recon_loss = F.binary_cross_entropy(atac_reconstructed_clamped, atac_expression_clamped)
        if torch.isnan(atac_recon_loss) or torch.isinf(atac_recon_loss):
            print("atac_recon_loss is nan or inf")
            print("atac_reconstructed min/max:", atac_reconstructed.min().item(), atac_reconstructed.max().item())
            print("atac_expression min/max:", atac_expression.min().item(), atac_expression.max().item())

        # Total loss (weights can be adjusted)
        total_loss = (
                rna_continuous_loss * self.rna_cont_weight +
                zinb_loss * self.zinb_weight +
                atac_recon_loss * self.atac_weight
        )

        # Store intermediate results for analysis
        self.binary_matrix = binary_matrix.detach()
        self.imputed_matrix = imputed_matrix.detach()

        return {
            "rna_continuous_loss": rna_continuous_loss,
            "zinb_loss": zinb_loss,
            "atac_reconstruction_loss": atac_recon_loss,
            "total_loss": total_loss
        }

    def get_imputed_matrix(self):
        """Get the imputed RNA expression matrix"""
        return self.imputed_matrix

    def get_binary_matrix(self):
        """Get the binarized expression matrix"""
        return self.binary_matrix


def build_rna_adjacency(adata, k=15, return_sparse=True):
    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    n_cells = X.shape[0]

    pearson_corr = np.corrcoef(X)
    sim_matrix = np.maximum(pearson_corr, 0)
    knn_indices = np.argsort(-sim_matrix, axis=1)[:, 1:k + 1]

    row_indices = np.repeat(np.arange(n_cells), k)
    col_indices = knn_indices.flatten()
    weights = sim_matrix[row_indices, col_indices]

    adj_matrix = csr_matrix((weights, (row_indices, col_indices)), shape=(n_cells, n_cells))
    adj_matrix = adj_matrix.maximum(adj_matrix.transpose())
    adj_matrix.setdiag(0)

    if not return_sparse:
        adj_matrix = adj_matrix.toarray()

    min_val = adj_matrix.min() if return_sparse else adj_matrix.data.min()
    max_val = adj_matrix.max() if return_sparse else adj_matrix.data.max()
    print(f"RNA adjacency matrix shape: {adj_matrix.shape} (cells×cells), value range: [{min_val:.4f}, {max_val:.4f}]")
    return adj_matrix


def preprocess_gene_expression(adata, num_hvg=2000):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=num_hvg, flavor="seurat")
    return adata[:, adata.var["highly_variable"]]


def process_activity_matrix(activity_matrix, top_n=2000, k_for_knn=20, return_sparse=True):
    cell_gene_matrix = activity_matrix.T
    cell_gene_matrix.columns = cell_gene_matrix.columns.astype(str)  # force column names to string
    scaler = MinMaxScaler()
    normalized_matrix = pd.DataFrame(
        scaler.fit_transform(cell_gene_matrix),
        columns=cell_gene_matrix.columns
    )

    gene_variances = normalized_matrix.var(axis=0)
    top_genes = gene_variances.nlargest(top_n).index
    filtered_matrix = cell_gene_matrix[top_genes]

    sim_matrix = cosine_similarity(filtered_matrix)
    knn_indices = np.argsort(-sim_matrix, axis=1)[:, 1:k_for_knn + 1]

    n_cells = filtered_matrix.shape[0]
    row_indices = np.repeat(np.arange(n_cells), k_for_knn)
    col_indices = knn_indices.flatten()
    weights = sim_matrix[row_indices, col_indices]

    adj_matrix = csr_matrix((weights, (row_indices, col_indices)), shape=(n_cells, n_cells))
    adj_matrix = adj_matrix.maximum(adj_matrix.transpose())
    adj_matrix.setdiag(0)

    if not return_sparse:
        adj_matrix = adj_matrix.toarray()

    print(f"ATAC adjacency matrix shape: {adj_matrix.shape} (cells×cells)")
    print(f"Filtered feature matrix shape: {filtered_matrix.shape} (cells×genes)")
    return adj_matrix, filtered_matrix


class ZINBLossForContinuous(nn.Module):
    """
    ZINB loss for continuous RNA data
    1. Convert continuous values to binary expression matrix (0/1)
    2. Use ZINB loss to train the binary matrix
    3. Estimate imputed values from ZINB parameters
    """

    def __init__(self, eps=1e-8, zero_threshold=0.01):
        super().__init__()
        self.eps = eps
        self.zero_threshold = zero_threshold  # threshold to decide zero

    def binarize_expression(self, x):
        """Convert continuous expression values to binary matrix (0 = not expressed, 1 = expressed)"""
        # The threshold can be adjusted based on data characteristics, e.g., median or quantile
        return (x > self.zero_threshold).float()

    def forward(self, continuous_data, mean_pred, pi_pred, theta=10.0):
        """
        Args:
            continuous_data: continuous RNA expression matrix (batch_size, num_genes)
            mean_pred: predicted mean of Negative Binomial distribution (batch_size, num_genes)
            pi_pred: predicted zero‑inflation probability (batch_size, num_genes)
            theta: dispersion parameter (can be fixed or learnable)

        Returns:
            loss: ZINB loss value
            binary_matrix: binarized expression matrix
            imputed_matrix: imputed matrix based on ZINB parameters
        """
        # 1. Generate binary expression matrix
        binary_matrix = self.binarize_expression(continuous_data)

        # 2. Compute ZINB loss (for binary matrix)
        loss = self._zinb_loss(binary_matrix, mean_pred, pi_pred, theta)

        # 3. Generate imputed matrix (combining ZINB parameters)
        imputed_matrix = self._generate_imputed_matrix(
            binary_matrix, mean_pred, pi_pred, continuous_data
        )

        return loss, binary_matrix, imputed_matrix

    def _zinb_loss(self, x_binary, mean, pi, theta):
        """Compute negative log‑likelihood of zero‑inflated Negative Binomial distribution"""
        # Ensure numerical stability
        mean = torch.clamp(mean, min=self.eps)
        pi = torch.clamp(pi, min=self.eps, max=1 - self.eps)

        # Convert theta to tensor with same shape as mean
        theta_tensor = torch.ones_like(mean) * theta

        # Negative Binomial log‑likelihood
        # For binary data, we only care about 0 and 1 cases
        # When x=0: from zero‑inflation part or NB part being 0
        # When x=1: from NB part being 1

        # Compute probability mass function of Negative Binomial
        log_prob_zero_nb = self._neg_binomial_log_prob(0, mean, theta_tensor)
        log_prob_one_nb = self._neg_binomial_log_prob(1, mean, theta_tensor)

        # For binarised continuous data, we redefine:
        # P(obs=0) = pi + (1-pi) * p_nb(0)
        # P(obs=1) = (1-pi) * p_nb(>=1) ≈ (1-pi) * (1 - p_nb(0))

        # NB: p(x>=1) = 1 - p(x=0)
        prob_zero_nb = torch.exp(log_prob_zero_nb)
        prob_at_least_one_nb = 1 - prob_zero_nb

        # ZINB mixture model log‑likelihood
        prob_zero = pi + (1 - pi) * prob_zero_nb
        prob_one = (1 - pi) * prob_at_least_one_nb

        # Add small epsilon to avoid log(0)
        prob_zero = torch.clamp(prob_zero, min=self.eps, max=1 - self.eps)
        prob_one = torch.clamp(prob_one, min=self.eps, max=1 - self.eps)

        # Log‑likelihood
        log_likelihood = x_binary * torch.log(prob_one) + (1 - x_binary) * torch.log(prob_zero)

        # Negative log‑likelihood loss
        return -torch.mean(log_likelihood)

    def _neg_binomial_log_prob(self, k, mu, theta):
        """Log‑probability of Negative Binomial distribution"""
        # Convert k to tensor, ensuring same device and dtype as mu and theta
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(k, device=mu.device, dtype=mu.dtype)

        log_theta = torch.log(theta)
        log_mu = torch.log(mu + self.eps)
        log_theta_mu = torch.log(theta + mu + self.eps)

        # Use log‑gamma to avoid numerical overflow
        # Note: when k is scalar, ensure shape broadcasting
        if k.dim() == 0:  # k is scalar
            k_expanded = k.expand_as(theta)
            term1 = torch.lgamma(k_expanded + theta) - torch.lgamma(theta) - torch.lgamma(k_expanded + 1)
        else:  # k is already a tensor
            term1 = torch.lgamma(k + theta) - torch.lgamma(theta) - torch.lgamma(k + 1)

        term2 = theta * (log_theta - log_theta_mu)
        term3 = k * (log_mu - log_theta_mu)

        return term1 + term2 + term3

    def _generate_imputed_matrix(self, binary_matrix, mean_pred, pi_pred, original_data):
        """
        Generate imputed matrix based on ZINB parameters
        Strategy:
        1. For positions where original is 0 but ZINB predicts expression, use mean_pred
        2. For expressed positions, combine original value and ZINB prediction
        """
        # Compute expression probability (ignoring zero‑inflation)
        expression_prob = 1 - pi_pred

        # Generate imputed matrix
        imputed = torch.zeros_like(original_data)

        # Case 1: originally expressed positions
        expressed_mask = binary_matrix > 0
        # Keep original expression values, but can incorporate ZINB smoothing
        imputed[expressed_mask] = original_data[expressed_mask]

        # Case 2: originally not expressed but predicted to be expressed
        # Threshold: expression_prob > 0.5 and pi_pred < 0.5
        should_express_mask = (expression_prob > 0.5) & (pi_pred < 0.5) & (~expressed_mask)
        # Impute with predicted mean
        imputed[should_express_mask] = mean_pred[should_express_mask]

        # Case 3: other non‑expressed positions set to 0
        imputed[~expressed_mask & ~should_express_mask] = 0

        return imputed


def plot_loss_curves(loss_history, save_path, loss_keys=None, labels=None, figsize=(16, 12)):
    """
    Plot convergence curves for multiple loss items (one subplot per loss)

    Parameters:
        loss_history: dict, must contain 'epoch' key and list for each loss key
        save_path: str, path to save the figure
        loss_keys: list, keys of losses to plot, e.g., ['rna_continuous_loss', 'zinb_loss', ...]
        labels: list, legend labels for each loss, length must match loss_keys
        figsize: tuple, overall figure size
    """
    if loss_keys is None:
        # Default plot for four losses (based on your model)
        loss_keys = ['rna_continuous_loss', 'zinb_loss', 'atac_reconstruction_loss', 'total_loss']
    if labels is None:
        # Default labels (can be customized)
        labels = ['RNA Continuous Loss', 'ZINB Loss', 'ATAC Reconstruction Loss', 'Total Loss']

    # Ensure keys exist
    for key in loss_keys:
        if key not in loss_history:
            raise KeyError(f"Loss key '{key}' not found in loss_history")

    epochs = loss_history['epoch']
    n_plots = len(loss_keys)

    # Automatically determine subplot layout (as close to square as possible)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # If only one subplot, axes might not be an array; handle uniformly
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Colors and markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', '*', 'p']

    for idx, (key, label) in enumerate(zip(loss_keys, labels)):
        ax = axes[idx]
        ax.plot(epochs, loss_history[key],
                label=label,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                markersize=4,
                linewidth=2.5)
        ax.set_title(f'{label} vs Epochs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots if any
    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    # Save image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # close figure to free memory
    print(f"Loss curves saved to: {save_path}")


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Read RNA data
    data = pd.read_csv('E:/jsj/user03/scATAC_data/X/rna_counts_filtered_unique.csv', index_col=0)
    data_filled = data.fillna(1)
    adata = sc.AnnData(data_filled.T)

    # RNA preprocessing
    adata_hvg = preprocess_gene_expression(adata)
    rna_features = adata_hvg.X.toarray()
    rna_adj = build_rna_adjacency(adata_hvg, k=15)
    rna_genes = [f"RNA_{gene}" for gene in adata_hvg.var_names.tolist()]
    rna_feature_cols = rna_features.shape[1]  # directly get actual number of RNA features
    print(f"RNA feature count: {rna_feature_cols}, gene names count: {len(rna_genes)}")  # useful for consistency check

    # Read ATAC data
    activity_df = pd.read_csv('E:/jsj/user03/scATAC_data/X/atac_activity.csv', index_col=0)
    activity_df.columns = activity_df.columns.astype(str)
    atac_adj, atac_features = process_activity_matrix(activity_df, top_n=2000, k_for_knn=15, return_sparse=True)
    print("Length of activity_df.index:", len(activity_df.index))

    # Check length of atac_features.columns
    print("Length of atac_features.columns:", len(atac_features.columns))
    atac_genes = [f"ATAC_{gene}" for gene in atac_features.columns]

    print("Length of atac_genes:", len(atac_genes))
    # Convert to numpy array
    atac_features = atac_features.values
    # Concatenate features
    combined_features = np.hstack((rna_features, atac_features))
    all_genes = rna_genes + atac_genes
    print(f"Combined feature shape: {combined_features.shape}")

    # Process labels
    true_labels = pd.read_csv("E:/jsj/user03/scATAC_data/X/clusters.csv")
    true_cluster = true_labels['Cluster']

    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_cluster)
    # Convert to tensors
    rna_adj_dense = rna_adj.toarray() if isinstance(rna_adj, csr_matrix) else rna_adj
    atac_adj_dense = atac_adj.toarray() if isinstance(atac_adj, csr_matrix) else atac_adj
    A1_tensor = torch.tensor(rna_adj_dense, dtype=torch.float32).to(device)
    A2_tensor = torch.tensor(atac_adj_dense, dtype=torch.float32).to(device)
    X_combined_tensor = torch.tensor(combined_features, dtype=torch.float32).to(device)
    rna_tensor = torch.tensor(rna_features, dtype=torch.float32).to(device)
    atac_tensor = torch.tensor(atac_features, dtype=torch.float32).to(device)
    # Define model

    num_classes = len(np.unique(true_labels_encoded))
    # You can adjust the number of input features as needed
    num_rna_features = rna_features.shape[1]  # number of RNA features
    num_atac_features = atac_features.shape[1]  # number of ATAC features
    # Instantiate model
    model = MultiModalAutoencoderWithZINB(
        num_rna_features, num_atac_features,
        vae_latent_dim=256, gcn_latent_dim=128,
        rna_cont_weight=0.5, zinb_weight=0.5, atac_weight=0.2,
        num_classes=num_classes
    ).to(device)
    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=0.001)  # use AdamW optimizer


    # Define training function
    def train(model, rna_tensor, atac_tensor, A1_tensor, A2_tensor, true_labels_encoded, optimizer, device, epochs=800,
              save_best_labels=True, record_loss=True):
        model.train()
        best_ari = -1.0
        best_labels = None
        true_labels_tensor = torch.tensor(true_labels_encoded, dtype=torch.long, device=device)

        loss_history = {
            'epoch': [],
            'rna_continuous_loss': [],
            'zinb_loss': [],
            'atac_reconstruction_loss': [],
            'total_loss': []
        }

        for epoch in range(epochs):
            optimizer.zero_grad()
            rna_continuous_recon, rna_mean, rna_pi, atac_reconstructed, losses, fused_embedding = model(
                rna_tensor, atac_tensor, A1_tensor, A2_tensor
            )
            total_loss = losses["total_loss"]

            total_loss.backward()
            optimizer.step()
            if record_loss:
                loss_history['epoch'].append(epoch + 1)
                loss_history['rna_continuous_loss'].append(losses['rna_continuous_loss'].item())
                loss_history['zinb_loss'].append(losses['zinb_loss'].item())
                loss_history['atac_reconstruction_loss'].append(losses['atac_reconstruction_loss'].item())
                loss_history['total_loss'].append(total_loss.item())
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    # Use fused embedding for clustering evaluation
                    latent = fused_embedding.detach().cpu().numpy()
                    kmeans = KMeans(n_clusters=len(np.unique(true_labels_encoded)), random_state=0)
                    cluster_labels = kmeans.fit_predict(latent)
                    ari = adjusted_rand_score(true_labels_encoded, cluster_labels)
                    print(f"Epoch [{epoch + 1}/{epochs}], Total Loss: {total_loss.item():.4f}, ")
                    # Update best record
                    if ari > best_ari:
                        best_ari = ari
                        best_labels = cluster_labels.copy()

        # Output best ARI after training
        print(f"\n=== Training Complete ===")
        print(f"Best ARI: {best_ari:.4f}")

        if save_best_labels and best_labels is not None:
            # Save best clustering labels as CSV
            df = pd.DataFrame(best_labels, columns=['Cluster'])
            df.to_csv('E:/jsj/user03/scATAC_data/X/best_cluster_labels3.csv', index=False)
            print("Best cluster labels saved to 'best_cluster_labels.csv'")

        return best_ari, loss_history


    # Start training
    # model.load_state_dict(torch.load("E:/jsj/user03/scATAC_data/human_PD_10x/trained_model_weights2.pth"))
    best_ari, loss_history = train(model, rna_tensor, atac_tensor, A1_tensor, A2_tensor, true_labels_encoded, optimizer, device, epochs=800, record_loss=True)
    torch.save(model.state_dict(), "E:/jsj/user03/scATAC_data/X/trained_model_weights2.pth")
    plot_loss_curves(
        loss_history,
        save_path="E:/jsj/user03/scATAC_data/X/loss_convergence_curve.png",
        loss_keys=['rna_continuous_loss', 'zinb_loss', 'atac_reconstruction_loss', 'total_loss'],
        labels=['RNA Continuous Loss', 'ZINB Loss', 'ATAC Reconstruction Loss', 'Total Loss'],
        figsize=(16, 12)
    )
    # ===================== After training, add the following code =====================
    # Ensure using the trained model (if re‑running this part, load weights first)
    # Model already in memory, set to evaluation mode
    model.eval()

    # Forward pass again to get fused embedding (without computing loss)
    with torch.no_grad():
        rna_continuous_recon, rna_mean, rna_pi, atac_reconstructed, losses, fused_embedding = model(
            rna_tensor, atac_tensor, A1_tensor, A2_tensor
        )
    latent_embedding = fused_embedding.cpu().numpy()  # shape (n_cells, latent_dim)

    # Use best clustering number (true number of classes) for KMeans clustering
    n_clusters = len(np.unique(true_labels_encoded))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_embedding)

    # Read true cell type names (original names, not encoded)
    true_celltypes = true_labels['CellType'].values  # ensure order matches cells

    # ---------- Hungarian algorithm to match clusters with true cell types ----------
    # 1. Prepare matching data
    match_data = pd.DataFrame({
        'true_celltype': true_celltypes,
        'pred_cluster': cluster_labels
    })

    # 2. Compute confusion matrix
    unique_celltypes = match_data['true_celltype'].unique()
    unique_clusters = match_data['pred_cluster'].unique()
    cm = confusion_matrix(
        match_data['true_celltype'],
        match_data['pred_cluster'],
        labels=unique_celltypes
    )

    # 3. Hungarian algorithm for optimal matching (maximise counts)
    from scipy.optimize import linear_sum_assignment

    row_indices, col_indices = linear_sum_assignment(cm, maximize=True)

    # 4. Build mapping dictionary from cluster to cell type
    cluster_to_celltype = {}
    for row, col in zip(row_indices, col_indices):
        cluster = unique_clusters[col]
        celltype = unique_celltypes[row]
        cluster_to_celltype[cluster] = celltype

    # 5. Handle any unmatched clusters (if present)
    for cluster in unique_clusters:
        if cluster not in cluster_to_celltype:
            cluster_mask = match_data['pred_cluster'] == cluster
            top_celltype = match_data[cluster_mask]['true_celltype'].mode().values[0]
            cluster_to_celltype[cluster] = top_celltype

    # 6. Convert each cell's cluster label to true cell type
    mapped_celltypes = [cluster_to_celltype[cluster] for cluster in cluster_labels]
    print("Mapping completed, cluster‑to‑celltype mapping:")
    for cluster, celltype in cluster_to_celltype.items():
        print(f"Cluster {cluster} → {celltype}")

    # ---------- t‑SNE dimensionality reduction and visualisation ----------
    print("Starting t‑SNE visualisation...")
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    tsne = TSNE(n_components=2, random_state=12, perplexity=30)
    tsne_embeddings = tsne.fit_transform(latent_embedding)

    # Custom colour palette (same as previous code)
    custom_palette = ['#ecd452', '#e76254', '#44757a', '#c6a468', '#6D4E7E',
                      '#2e5496', '#299d8f', '#4994c4', '#8f6d5f', '#FAC050',
                      '#77B473', '#CBC2D7', '#D28450', '#DCC2AB', '#90C3B6',
                      '#FDC192', '#3854A6', '#727174', '#F3756D', '#FAF49B',
                      '#9172DB', '#5A9', '#B5D8FE', '#FF9F9B', '#DDBEA9']

    labels_to_plot = mapped_celltypes
    unique_labels = np.unique(labels_to_plot)
    num_labels = len(unique_labels)

    # Extend colours if insufficient
    if len(custom_palette) < num_labels:
        from matplotlib import cm

        additional_colors = [cm.tab20(i) for i in range(num_labels - len(custom_palette))]
        custom_palette.extend(additional_colors)
        print(f"Colour list extended to {num_labels} colours")

    color_map = dict(zip(unique_labels, custom_palette[:num_labels]))
    color_values = np.array([color_map[label] for label in labels_to_plot])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=color_values,
        s=15,
        alpha=1.0
    )
    plt.title('t‑SNE Visualisation (True Cell Types)', fontsize=15)
    plt.xlabel('t‑SNE Component 1', fontsize=12)
    plt.ylabel('t‑SNE Component 2', fontsize=12)

    handles = [mpatches.Patch(color=color_map[label], label=label) for label in unique_labels]
    ax.legend(handles=handles, title="Cell Type", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()
    tsne_save_path = "E:/jsj/user03/scATAC_data/X/tsne/tsne_new.png"
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    print(f"t‑SNE visualisation saved to: {tsne_save_path}")

    # Optionally save mapped labels for further analysis
    mapped_df = pd.DataFrame({'original_cluster': cluster_labels, 'mapped_celltype': mapped_celltypes})
    mapped_df.to_csv(
        "E:/jsj/user03/scATAC_data/X/mapped_celltypes.csv",
        index=False)
    print("Mapped cell type labels saved.")

