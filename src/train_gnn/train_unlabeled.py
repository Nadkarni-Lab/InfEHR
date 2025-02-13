# Standard library imports
import os
import sys
import re
import gc
import json
import random
import logging
import shutil
import pickle
import argparse
import subprocess
from typing import Callable
from datetime import datetime
from collections import Counter


# Environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Third-party imports
import numpy as np
import networkx as nx
import platform
from tqdm import tqdm
import joblib

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import (
    CyclicLR,
    ReduceLROnPlateau
)

# PyTorch Geometric imports
import torch_geometric
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import (
    ASAPooling, 
    GATConv, 
    GCNConv,
    SAGPooling,
    PANPooling,
    MessagePassing,
    MLPAggregation,
    AttentionalAggregation,
    graclus,
    avg_pool_x,
    avg_pool_neighbor_x,
    global_mean_pool, 
    global_max_pool, 
    global_add_pool,
    Linear,
    aggr
)
from torch_geometric.utils import (
    from_networkx,
    to_dense_adj,
    dense_to_sparse,
    k_hop_subgraph,
    add_self_loops,
    degree,
    sort_edge_index,
    negative_sampling,
    batched_negative_sampling
)
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_cluster import graclus_cluster


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train GNN model with VicReg and custom loss functions')
    
    # Required path arguments (keep existing)
    parser.add_argument('--log-dir', required=True,
                      help='Directory for saving log files')
    parser.add_argument('--model-save-dir', required=True,
                      help='Directory for saving model checkpoints')
    parser.add_argument('--graph-dir', required=True,
                      help='Directory containing graph data files')
    parser.add_argument('--graph-mapping', required=True,
                      help='Path to JSON file containing MRN to filename mappings')
    
    # VicReg Loss weights
    parser.add_argument('--sim-weight', type=float, default=0.5,
                      help='Weight for similarity loss')
    parser.add_argument('--var-weight', type=float, default=3.0,
                      help='Weight for variance loss')
    parser.add_argument('--cov-weight', type=float, default=2.0,
                      help='Weight for covariance loss')
    parser.add_argument('--mi-weight', type=float, default=1.0,
                      help='Weight for mutual information loss')
    
    # Model architecture (keep existing and add)
    parser.add_argument('--input-dim', type=int, default=464,
                      help='Input dimension for the model')
    parser.add_argument('--hidden-dim', type=int, default=256,
                      help='Hidden dimension for the model')
    parser.add_argument('--embedding-dim', type=int, default=128,
                      help='Embedding dimension for the model')
    parser.add_argument('--num-heads', type=int, default=2,
                      help='Number of attention heads')
    parser.add_argument('--pool-ratio', type=float, default=0.2,
                      help='Pooling ratio for ASAPooling')
    
    # Training configuration (update existing)
    parser.add_argument('--epochs', type=int, default=1000,  # Changed from 150 to 1000
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,  # Changed from 0.001
                      help='Learning rate for optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay for optimizer')
    parser.add_argument('--accumulation-steps', type=int, default=64,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=0,
                      help='Number of data loading workers')
    parser.add_argument('--log-frequency', type=int, default=20,
                      help='How often to log model outputs during training')
    
    # Additional options
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use for training')
    parser.add_argument('--skip-indices', type=str, 
                      default='8,189,359,534,548,631,700,753,1033,1178,1605,1778,2050,2136,2291',
                      help='Comma-separated list of indices to skip for optimal memory stability, depending on GPU state they can be included')
    
    args = parser.parse_args()
    args.skip_indices = {int(idx) for idx in args.skip_indices.split(',')}
    
    return args

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_system_info():
    """Collect system and environment information."""
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
    
    return info

def get_gpu_info():
    """Get NVIDIA GPU information using nvidia-smi."""
    try:
        output = subprocess.run(
            ['nvidia-smi'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        return output.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running nvidia-smi: {e.stderr}"
    except FileNotFoundError:
        return "nvidia-smi not found in PATH"

def log_system_info(logger):
    """Log all system information."""
    logger.info("========== System Information ==========")
    
    # Get system info
    sys_info = get_system_info()
    for key, value in sys_info.items():
        logger.info(f"{key}: {value}")
    
    # Get GPU info
    logger.info("\n========== GPU Information ==========")
    gpu_info = get_gpu_info()
    logger.info(f"\n{gpu_info}")
    
    # Log CUDA specific details if available
    if torch.cuda.is_available():
        logger.info("\n========== CUDA Information ==========")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"CUDA architecture list: {torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else 'Not available'}")

def verify_cuda_installation(logger):
    """Detailed verification of CUDA setup."""
    import torch
    import os
    
    logger.info("\n========== CUDA Installation Verification ==========")
    
    # Check environment variables
    cuda_home = os.environ.get('CUDA_HOME')
    cuda_path = os.environ.get('CUDA_PATH')
    logger.info(f"CUDA_HOME: {cuda_home}")
    logger.info(f"CUDA_PATH: {cuda_path}")
    
    # PyTorch CUDA info
    logger.info(f"\nPyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("\nPotential CUDA issues:")
        logger.info("1. PyTorch not installed with CUDA support")
        logger.info("2. CUDA toolkit not properly installed")
        logger.info("3. Environment variables not set correctly")
        logger.info("\nTry running:")
        logger.info("conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia --force-reinstall")
    
    # Try CUDA tensor creation
    try:
        x = torch.tensor([1.0, 2.0])
        if torch.cuda.is_available():
            x = x.cuda()
            logger.info("\nSuccessfully created CUDA tensor!")
    except Exception as e:
        logger.error(f"\nError creating CUDA tensor: {str(e)}")

def get_timestamp():
    """Generate a clean timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS 


def validate_data_loader(model, data_loader, logger, device='cuda'):
    model.eval()
    total_graphs = len(data_loader.dataset)
    failed_indices = []

    if torch.cuda.is_available() and device == 'cuda':
        model.to(device)
    else:
        device = 'cpu'
        model.to(device)
        logger.info('On CPU')
    
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            try:
                # Optional: torch.cuda.empty_cache()  # Clear cache before each batch
                batch = batch.to(device)
                output = model(batch)
                pseudo_loss = output.sum()

                if torch.isnan(pseudo_loss).any() or torch.isinf(pseudo_loss).any():
                    raise ValueError(f"Invalid pseudo_loss value at idx {idx}")
        
                # Validation with sizes
                assert batch.x is not None, f"Missing node features at idx {idx}"
                assert batch.edge_index is not None, f"Missing edges at idx {idx}"
                assert batch.y is not None, f"Missing label at idx {idx}"
                
                logger.info(f"Graph {idx+1}/{total_graphs} - Nodes: {batch.x.shape[0]}, Edges: {batch.edge_index.shape[1]}")
                
            except Exception as e:
                failed_indices.append((idx, str(e)))
                logger.error(f"Graph {idx} failed: {str(e)}", exc_info=True)
                continue  # Continue to next graph
                
    if failed_indices:
        failure_details = '\n'.join([f"Graph {idx}: {err}" for idx, err in failed_indices])
        logger.error(f"Failed graphs:\n{failure_details}")
        raise ValueError(f"{len(failed_indices)} graphs failed validation")
    else:
        logger.info(f"All {total_graphs} graphs validated successfully")


class MemoryEfficientGraphDataset(Dataset):
    """
    A memory-efficient dataset for loading large graphs one at a time from disk.
    
    Parameters
    ----------
    root_dir : str
        Base directory containing the graphs
    graph_mapping_file : str
        Path to JSON file containing MRN to filename mappings
    transform : callable, optional
        Transform to be applied to each graph
    required_attributes : list, optional
        List of required attributes each graph should have (default: ['x', 'edge_index', 'y'])
    """
    
    def __init__(
        self,
        root_dir,
        graph_mapping_file,
        transform=None,
        required_attributes=['x', 'edge_index', 'y']
    ):
        # Store root directory without calling super() yet
        self.root = root_dir
        
        # Now call super with just the transform
        super().__init__(root_dir, transform)
        
        self._validate_paths(root_dir, graph_mapping_file)
        self.required_attributes = required_attributes
        self.mrn_to_file = self._load_mapping_file(graph_mapping_file)
        self.mrns = list(self.mrn_to_file.keys())
        
        self._validate_dataset()
        logger.info(f"Dataset initialized with {len(self.mrns)} graphs")
    
    def _validate_paths(self, root_dir, graph_mapping_file):
        """Validate that all required paths exist."""
        if not os.path.exists(root_dir):
            raise ValueError(f"root_dir does not exist: {root_dir}")
        if not os.path.exists(graph_mapping_file):
            raise ValueError(f"graph_mapping_file does not exist: {graph_mapping_file}")
    
    def _load_mapping_file(self, graph_mapping_file):
        """Load and validate the MRN to filename mapping."""
        try:
            with open(graph_mapping_file, 'r') as f:
                mapping = json.load(f)
            
            if not mapping:
                raise ValueError("Mapping file is empty")
            return mapping
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in mapping file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading mapping file: {str(e)}")
    
    def _validate_dataset(self):
        """Validate dataset initialization and first graph loading."""
        if len(self.mrns) == 0:
            raise ValueError("No graphs found in mapping file")
        
        # Try loading the first graph to verify everything works
        try:
            self.get(0)
        except Exception as e:
            raise ValueError(f"Failed to load first graph: {str(e)}")
    
    def _validate_graph_attributes(self, data, mrn):
        """Verify that a graph has all required attributes."""
        missing_attrs = [attr for attr in self.required_attributes if not hasattr(data, attr)]
        if missing_attrs:
            raise ValueError(
                f"Graph {mrn} missing required attributes: {', '.join(missing_attrs)}"
            )
    
    def len(self):
        """Return the number of graphs in the dataset."""
        return len(self.mrns)
    
    def get(self, idx):
        """
        Load a single graph by index.
        
        Parameters
        ----------
        idx : int
            Index of the graph to load
            
        Returns
        -------
        torch_geometric.data.Data
            The loaded graph data object
        """
        if idx >= len(self.mrns):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.mrns)}")
        
        mrn = self.mrns[idx]
        # Don't join with root directory as the path in the JSON already includes it
        filepath = os.path.join(self.root, os.path.basename(self.mrn_to_file[mrn]))
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        try:
            data = torch.load(filepath)
            self._validate_graph_attributes(data, mrn)
            
            if self.transform is not None:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Error loading graph {mrn}: {str(e)}")


# ------------------------------- Model Definition ---------------------------- # 


class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class GNNVicRegWithMISample(torch.nn.Module):
    def __init__(self, input_dim=464, hidden_dim=256, embedding_dim=128, 
                 num_heads=2, pool_ratio=0.2):
        super().__init__()
        # Existing VicReg components
        self.pool1 = ASAPooling(input_dim, ratio=pool_ratio)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, embedding_dim, concat=True, heads=num_heads)
        self.residual = Linear(hidden_dim * num_heads, embedding_dim * num_heads)
        self.dim_reduce = Linear(embedding_dim * num_heads, embedding_dim)
        
        self.projector1 = MLPProjector(input_dim, hidden_dim, embedding_dim)
        self.projector2 = MLPProjector(embedding_dim, hidden_dim, embedding_dim)

        # Add MI scoring matrix as a model parameter
        self.W_mi = nn.Parameter(torch.randn(embedding_dim, embedding_dim))

    def compute_mi_score(self, local_features, global_features):
        # Add dimension check and reshape
        if local_features.dim() == 1:
            local_features = local_features.unsqueeze(0)
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        
        local_proj = local_features @ self.W_mi  # Matrix multiplication
        scores = torch.sum(local_proj * global_features, dim=1)
        return scores


    def forward(self, x, edge_index, batch):
        view1 = self.projector1(x)
        
        # Ensure edge_index is valid before pooling
        max_index = x.size(0) - 1
        edge_index = edge_index.clamp(0, max_index)
        
        # Adjust pooling to match tensor dimensions
        x_pooled, edge_index_pooled, _, batch_pooled, _ = self.pool1(
            x, 
            edge_index,
            None,
            batch)  # Add explicit size parameter
        x_pooled = F.relu(self.conv1(x_pooled, edge_index_pooled))
        residual = self.residual(x_pooled)
        x_pooled = F.relu(self.conv2(x_pooled, edge_index_pooled)) + residual
        x_pooled = self.dim_reduce(x_pooled)
        view2 = self.projector2(x_pooled)
        
        return view1, view2, batch, batch_pooled, self.W_mi, edge_index_pooled


class VicRegMILossSingleGraphSample(nn.Module):
    def __init__(self, sim_weight=0.5, var_weight=3.0, cov_weight=2.0, mi_weight=1.0, eps=1e-4, logger=logger):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.mi_weight = mi_weight
        self.eps = eps
        self.logger = logger
        

    def off_diagonal(self, x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

    def variance_loss(self, view1, view2):
        std1 = torch.sqrt(view1.var(dim=0) + self.eps)
        std2 = torch.sqrt(view2.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))
        return std_loss

    def covariance_loss(self, view1, view2):
        view1 = view1 - view1.mean(dim=0)
        view2 = view2 - view2.mean(dim=0)
        
        N1 = view1.shape[0]
        N2 = view2.shape[0]
        D = view1.shape[1]
        
        cov1 = (view1.T @ view1) / (N1 - 1 + self.eps)
        cov2 = (view2.T @ view2) / (N2 - 1 + self.eps)
        
        cov_loss = (self.off_diagonal(cov1).pow(2).sum() / D + 
                     self.off_diagonal(cov2).pow(2).sum() / D)
        return cov_loss

    def mi_loss_single_graph(self, model, view, batch, edge_index):
        if batch.dim() > 1:
            batch = batch.squeeze()

        # Create temporary Data object for pooling
        temp_data = Data(x=view, edge_index=edge_index)
        pooled_data = avg_pool_neighbor_x(temp_data)
        pooled_view = pooled_data.x
        pooled_batch = batch[:pooled_view.size(0)]

        global_emb = global_mean_pool(pooled_view, pooled_batch)

        # Corruption
        corrupted_view = view.clone()
        window_size = max(2, int(view.size(0) * 0.1))

        for b in torch.unique(batch):
            batch_mask = (batch == b)
            nodes_to_corrupt = corrupted_view[batch_mask]
            for start_idx in range(0, len(nodes_to_corrupt), window_size):
                end_idx = min(start_idx + window_size, len(nodes_to_corrupt))
                perm = torch.randperm(end_idx - start_idx)
                nodes_to_corrupt[start_idx:end_idx] = nodes_to_corrupt[start_idx:end_idx][perm]
            corrupted_view[batch_mask] = nodes_to_corrupt

        # Pool corrupted view
        temp_corrupted_data = Data(x=corrupted_view, edge_index=edge_index)
        corrupted_pooled_data = avg_pool_neighbor_x(temp_corrupted_data)
        corrupted_pooled = corrupted_pooled_data.x
        corrupted_batch = batch[:corrupted_pooled.size(0)]

        corrupted_global = global_mean_pool(corrupted_pooled, corrupted_batch)

        pos_scores = model.compute_mi_score(pooled_view, global_emb)
        neg_scores = model.compute_mi_score(pooled_view, corrupted_global)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([
            torch.ones(pos_scores.size(0), device=scores.device),
            torch.zeros(neg_scores.size(0), device=scores.device)
        ])
        
        return F.binary_cross_entropy_with_logits(scores, labels)

    def forward(self, model, view1, view2, batch1, batch2, edge_index, edge_index_pooled):
        # Clamp values
        view1 = torch.clamp(view1, -1e6, 1e6)
        view2 = torch.clamp(view2, -1e6, 1e6)
        
        # VICReg losses use global_mean_pool as original
        view1_pooled = global_mean_pool(view1, batch1)
        view2_pooled = global_mean_pool(view2, batch2)
        
        sim_loss = F.mse_loss(view1_pooled, view2_pooled)
        var_loss = self.variance_loss(view1, view2)
        cov_loss = self.covariance_loss(view1, view2)

        mi_loss_view1 = self.mi_loss_single_graph(model, view1, batch1, edge_index)

        num_pooled_nodes = view2.size(0)
        max_edge_idx = edge_index_pooled.max().item()

        if max_edge_idx >= num_pooled_nodes:
            self.logger.info("ERROR: Pooled edges out of bounds!")

        mi_loss_view2 = self.mi_loss_single_graph(model, view2, batch2, edge_index_pooled)
        
        mi_loss = 0.5 * (mi_loss_view1 + mi_loss_view2)

        total_loss = (self.sim_weight * sim_loss +
                    self.var_weight * var_loss +
                    self.cov_weight * cov_loss +
                    self.mi_weight * mi_loss)

        return total_loss, {
            'sim_loss': sim_loss.item(),
            'var_loss': var_loss.item(),
            'cov_loss': cov_loss.item(),
            'mi_loss': mi_loss.item()
        }

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = get_timestamp()
    log_file_path = os.path.join(args.log_dir, f'gnn_exp_train_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log system information
    log_system_info(logger)
    verify_cuda_installation(logger)

    # Set device and seeds
    set_deterministic()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')

    try:
        # Initialize dataset
        dataset = MemoryEfficientGraphDataset(
            root_dir=args.graph_dir,
            graph_mapping_file=args.graph_mapping
        )
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Log dataset info
        first_batch = next(iter(train_loader))
        logger.info(f"Dataset initialized with shape: {first_batch.x.shape}")
        logger.info(f"Labels: {first_batch.y}")

        # Initialize model
        model = GNNVicRegWithMISample(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            pool_ratio=args.pool_ratio
        ).to(device)

        # Initialize criterion
        criterion = VicRegMILossSingleGraphSample(
            sim_weight=args.sim_weight,
            var_weight=args.var_weight,
            cov_weight=args.cov_weight,
            mi_weight=args.mi_weight,
            logger=logger
        ).to(device)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Training initialization
        logger.info("Starting training...")
        all_failed_indices = set()
        best_loss = float('inf')
        
        # Training loop
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            accumulated_loss = 0.0
            epoch_failed_indices = set()
            
            logger.info(f'Beginning Epoch {epoch + 1}')
            optimizer.zero_grad()

            for step, data in enumerate(train_loader):
                if step in args.skip_indices:
                    logger.info(f"Skipping problematic graph {step}")
                    continue

                try:
                    data = data.to(device)
                    view1, view2, batch1, batch2, W_mi, edge_index_pooled = model(
                        data.x, 
                        data.edge_index, 
                        data.batch
                    )
                    
                    loss, loss_components = criterion(
                        model=model,
                        view1=view1, 
                        view2=view2,
                        batch1=batch1, 
                        batch2=batch2,
                        edge_index=data.edge_index,
                        edge_index_pooled=edge_index_pooled
                    )
                    
                    scaled_loss = loss / args.accumulation_steps
                    scaled_loss.backward()

                    total_loss += loss.detach().item()
                    accumulated_loss += loss.detach().item()

                    if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                        avg_accumulated_loss = accumulated_loss / args.accumulation_steps
                        logger.info(f'Step {step + 1}: Average accumulated batch loss: {avg_accumulated_loss:.4f}')
                        logger.info(
                            f'Loss components - Sim: {loss_components["sim_loss"]:.4f}, '
                            f'Var: {loss_components["var_loss"]:.4f}, '
                            f'Cov: {loss_components["cov_loss"]:.4f}, '
                            f'MI: {loss_components["mi_loss"]:.4f}'
                        )
                        
                        accumulated_loss = 0.0
                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.cuda.synchronize()

                except (torch.cuda.OutOfMemoryError, Exception) as e:
                    epoch_failed_indices.add(step)
                    all_failed_indices.add(step)
                    logger.error(f'Error at epoch {epoch}, step {step}: {str(e)}')
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue

            if epoch_failed_indices:
                logger.warning(f'Failed indices in epoch {epoch}: {sorted(epoch_failed_indices)}')

            num_successful_steps = len(train_loader) - len(epoch_failed_indices)
            if num_successful_steps > 0:
                average_epoch_loss = total_loss / (num_successful_steps * args.accumulation_steps)
                logger.info(
                    f'Epoch {epoch} average loss: {average_epoch_loss:.4f} '
                    f'(computed over {num_successful_steps}/{len(train_loader)} steps)'
                )

                if average_epoch_loss < best_loss:
                    best_loss = average_epoch_loss
                    timestamp = get_timestamp()
                    model_filename = f'epoch_{epoch}_loss_{average_epoch_loss:.4f}_{timestamp}_ssl_model.pth'
                    model_save_path = os.path.join(args.model_save_dir, model_filename)
                    
                    logger.info(f'Saving new best model with average loss: {average_epoch_loss:.4f} at epoch {epoch+1}')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'timestamp': timestamp
                    }, model_save_path)

        logger.info(f"Training completed. Best loss: {best_loss}")
        if all_failed_indices:
            logger.info(f"Failed indices during training: {sorted(all_failed_indices)}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
