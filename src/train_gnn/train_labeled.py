# Standard library imports
import os
import sys
import json
import logging
import platform
import subprocess
from datetime import datetime
import argparse

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import ASAPooling, GATConv, global_mean_pool, Linear
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import numpy as np
from tqdm import tqdm

# Environment configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
EPSILON = 1e-11

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train GNN model with custom loss functions')
    
    # Required path arguments
    parser.add_argument('--log-dir', required=True,
                      help='Directory for saving log files')
    parser.add_argument('--model-save-dir', required=True,
                      help='Directory for saving model checkpoints')
    parser.add_argument('--graph-dir', required=True,
                      help='Directory containing graph data files')
    parser.add_argument('--graph-mapping', required=True,
                      help='Path to JSON file containing MRN to filename mappings')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=150,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
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
    
    # Model configuration
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
    parser.add_argument('--num-classes', type=int, default=2,
                      help='Number of output classes')
    
    # Loss function configuration
    parser.add_argument('--custom-loss-epoch', type=int, default=1,
                      help='Epoch to switch to custom loss function')
    parser.add_argument('--weight-lr', type=float, default=0.0001,
                      help='Learning rate for loss function weight network')
    
    # Additional options
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use for training')
    parser.add_argument('--skip-indices', type=str, default='8,189,359,534,548,631,700,753,1033,1178,1605,1778,2050,2136,2291',
                      help='Comma-separated list of indices to skip during training for memory stability, variations in GPU state may allow all indices to be used.')
    
    args = parser.parse_args()
    
    # Convert skip_indices from string to set of integers
    args.skip_indices = {int(idx) for idx in args.skip_indices.split(',')} 
    
    return args

# System information functions
def get_system_info():
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
    logger.info("========== System Information ==========")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\n========== GPU Information ==========")
    gpu_info = get_gpu_info()
    logger.info(f"\n{gpu_info}")
    
    if torch.cuda.is_available():
        logger.info("\n========== CUDA Information ==========")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"CUDA architecture list: {torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else 'Not available'}")

def verify_cuda_installation(logger):
    logger.info("\n========== CUDA Installation Verification ==========")
    
    cuda_home = os.environ.get('CUDA_HOME')
    cuda_path = os.environ.get('CUDA_PATH')
    logger.info(f"CUDA_HOME: {cuda_home}")
    logger.info(f"CUDA_PATH: {cuda_path}")
    
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
    
    try:
        x = torch.tensor([1.0, 2.0])
        if torch.cuda.is_available():
            x = x.cuda()
            logger.info("\nSuccessfully created CUDA tensor!")
    except Exception as e:
        logger.error(f"\nError creating CUDA tensor: {str(e)}")

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
                batch = batch.to(device)
                output = model(batch)
                pseudo_loss = output.sum()

                if torch.isnan(pseudo_loss).any() or torch.isinf(pseudo_loss).any():
                    raise ValueError(f"Invalid pseudo_loss value at idx {idx}")
        
                assert batch.x is not None, f"Missing node features at idx {idx}"
                assert batch.edge_index is not None, f"Missing edges at idx {idx}"
                assert batch.y is not None, f"Missing label at idx {idx}"
                
                logger.info(f"Graph {idx+1}/{total_graphs} - Nodes: {batch.x.shape[0]}, Edges: {batch.edge_index.shape[1]}")
                
            except Exception as e:
                failed_indices.append((idx, str(e)))
                logger.error(f"Graph {idx} failed: {str(e)}", exc_info=True)
                continue
                
    if failed_indices:
        failure_details = '\n'.join([f"Graph {idx}: {err}" for idx, err in failed_indices])
        logger.error(f"Failed graphs:\n{failure_details}")
        raise ValueError(f"{len(failed_indices)} graphs failed validation")
    else:
        logger.info(f"All {total_graphs} graphs validated successfully")

def test_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        sample_batch = sample_batch.to(device)
        
        try:
            output = model(sample_batch)
            logger.info(f"Model test successful!")
            logger.info(f"Input shape: {sample_batch.x.shape}")
            logger.info(f"Output shape: {output.shape}")
            return True
        except Exception as e:
            logger.info(f"Model test failed: {str(e)}")
            return False

# Dataset classes
class MemoryEfficientGraphDataset(Dataset):
    def __init__(self, root_dir, graph_mapping_file, transform=None, required_attributes=['x', 'edge_index', 'y']):
        self.root = root_dir
        super().__init__(root_dir, transform)
        
        self._validate_paths(root_dir, graph_mapping_file)
        self.required_attributes = required_attributes
        self.mrn_to_file = self._load_mapping_file(graph_mapping_file)
        self.mrns = list(self.mrn_to_file.keys())
        
        self._validate_dataset()
        logger.info(f"Dataset initialized with {len(self.mrns)} graphs")
    
    def _validate_paths(self, root_dir, graph_mapping_file):
        if not os.path.exists(root_dir):
            raise ValueError(f"root_dir does not exist: {root_dir}")
        if not os.path.exists(graph_mapping_file):
            raise ValueError(f"graph_mapping_file does not exist: {graph_mapping_file}")
    
    def _load_mapping_file(self, graph_mapping_file):
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
        if len(self.mrns) == 0:
            raise ValueError("No graphs found in mapping file")
        
        try:
            self.get(0)
        except Exception as e:
            raise ValueError(f"Failed to load first graph: {str(e)}")
    
    def _validate_graph_attributes(self, data, mrn):
        missing_attrs = [attr for attr in self.required_attributes if not hasattr(data, attr)]
        if missing_attrs:
            raise ValueError(
                f"Graph {mrn} missing required attributes: {', '.join(missing_attrs)}"
            )
    
    def len(self):
        return len(self.mrns)
    
    def get(self, idx):
        if idx >= len(self.mrns):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.mrns)}")
        
        mrn = self.mrns[idx]
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

class FilteredMemoryEfficientGraphDataset(MemoryEfficientGraphDataset):
    def __init__(self, root_dir, graph_mapping_file, exclude_indices, transform=None,
                 required_attributes=['x', 'edge_index', 'y']):
        self.filtered_mrns = None
        super().__init__(root_dir, graph_mapping_file, transform, required_attributes)
        
        excluded_mrns = [self.mrns[i] for i in exclude_indices]
        self.filtered_mrns = [mrn for mrn in self.mrns if mrn not in excluded_mrns]
        
        logger.info(f"Filtered dataset: {len(self.filtered_mrns)}/{len(self.mrns)} graphs")

    def len(self):
        return len(self.filtered_mrns)

    def get(self, idx):
        if not self.filtered_mrns:
            return super().get(idx)
        mrn = self.filtered_mrns[idx]
        return super().get(self.mrns.index(mrn))

# Model class
class GNNModelWithASAPoolingAndGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_heads=2, pool_ratio=0.5, num_classes=2):
        super(GNNModelWithASAPoolingAndGAT, self).__init__()
        
        self.pool1 = ASAPooling(input_dim, ratio=pool_ratio)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, embedding_dim, heads=num_heads)
        self.residual = Linear(hidden_dim * num_heads, embedding_dim * num_heads)
        self.fc_final = Linear(embedding_dim * num_heads, num_classes)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, data):
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'batch'):
            raise ValueError("Input data missing required attributes (x, edge_index, or batch)")
            
        if data.x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {data.x.size(-1)}")
            
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv1(x, edge_index))
        residual = self.residual(x)
        x = F.relu(self.conv2(x, edge_index)) + residual
        gmean_pool = global_mean_pool(x, batch)
        x = self.fc_final(gmean_pool)
        
        return x, gmean_pool

    def verify_input(self, data):
        if not hasattr(data, 'x'):
            raise ValueError("Input data missing node features (x)")
        if not hasattr(data, 'edge_index'):
            raise ValueError("Input data missing edge indices (edge_index)")
        if not hasattr(data, 'batch'):
            raise ValueError("Input data missing batch information (batch)")
            
        if data.x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {data.x.size(-1)}")

def initialize_model(input_dim, hidden_dim, embedding_dim, num_heads, pool_ratio, num_classes, device=None):
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'embedding_dim': embedding_dim,
        'num_heads': num_heads,
        'pool_ratio': pool_ratio,
        'num_classes': num_classes
    }
    model = GNNModelWithASAPoolingAndGAT(**model_params)
    if device is not None:
        model = model.to(device)
    return model

# Loss functions
def implied_log_posterior(log_preds, prior, eps=EPSILON):
    kp2 = log_preds.ndim
    dim = 0 if kp2 == 2 else (0, *range(2, kp2))

    denom = log_preds.logsumexp(dim=dim, keepdim=True)
    log_s = log_preds - denom
    r_prime = log_s.exp() * prior
    log_r_prime = r_prime.clamp(min=eps).log()
    return log_r_prime.log_softmax(1)

def RQ_loss(preds, prior, log_preds=False, eps=EPSILON):
    log_preds = preds if log_preds else preds.log_softmax(1)
    log_r = implied_log_posterior(log_preds, prior, eps=eps)
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
    return kl_loss(log_preds, log_r)

def QR_loss(preds, prior, log_preds=False, eps=EPSILON):
    log_preds = preds if log_preds else preds.log_softmax(1)
    log_r = implied_log_posterior(log_preds, prior, eps=eps)
    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
    return kl_loss(log_r, log_preds)

class LossWrapper:
    def __init__(self, name):
        self.name = name

    def _get_name(self):
        return self.name

    def maybe_one_hot_encode(self, preds, prior):
        if prior.ndim == preds.ndim - 1:
            prior = F.one_hot(prior, num_classes=preds.shape[1])
        elif prior.ndim != preds.ndim:
            raise ValueError(f"Got incompatible shapes, prior: {prior.shape} and preds: {preds.shape}.")
        return prior

class RQ(LossWrapper):
    def __init__(self, log_preds=False, eps=EPSILON):
        super().__init__('RQ')
        self.log_preds = log_preds
        self.eps = eps

    def __call__(self, preds, prior):
        prior = self.maybe_one_hot_encode(preds, prior)
        return RQ_loss(preds, prior, log_preds=self.log_preds, eps=self.eps)

class QR(LossWrapper):
    def __init__(self, log_preds=False, eps=EPSILON):
        super().__init__('QR')
        self.log_preds = log_preds
        self.eps = eps

    def __call__(self, preds, prior):
        prior = self.maybe_one_hot_encode(preds, prior)
        return QR_loss(preds, prior, log_preds=self.log_preds, eps=self.eps)

class VirtualBatchMIWeightedQR(LossWrapper):
    def __init__(self, accumulation_steps, feature_dim, weight_lr=0.0001, log_preds=False, eps=EPSILON):
        super().__init__('MI_QR')
        self.log_preds = log_preds
        self.eps = eps
        self.qr_loss = RQ(log_preds=log_preds, eps=eps)
        self.accumulation_steps = accumulation_steps
        
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim, 100),  
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        ).to('cuda')
        self.weight_optimizer = optim.Adam(self.weight_net.parameters(), lr=weight_lr)
        
        self.reset_storage()

    def reset_storage(self):
        self.stored_preds = []
        self.stored_priors = []
        self.stored_features = []
        self.batch_count = 0
        torch.cuda.empty_cache()

    def __call__(self, preds, prior, penultimate_features):
        prior = self.maybe_one_hot_encode(preds, prior)
        
        self.stored_preds.append(preds.detach())
        self.stored_priors.append(prior.detach())
        self.stored_features.append(penultimate_features.detach())
        self.batch_count += 1

        if self.batch_count >= self.accumulation_steps:
            try:
                all_preds = torch.cat(self.stored_preds, dim=0)
                all_priors = torch.cat(self.stored_priors, dim=0)
                all_features = torch.cat(self.stored_features, dim=0)
                
                weights = self.weight_net(all_features).squeeze()
                weights = F.softmax(weights, dim=0)
                
                base_loss = self.qr_loss(all_preds, all_priors)
                loss = (weights * base_loss).mean() / self.accumulation_steps
                
                return loss
            finally:
                self.reset_storage()
        
        return self.qr_loss(preds, prior) / self.accumulation_steps

    def cleanup(self):
        self.reset_storage()
        torch.cuda.empty_cache()

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Logger setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(args.log_dir, f'gnn_exp_train_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # System info logging
    log_system_info(logger)
    verify_cuda_installation(logger)

    # Set device and seeds
    set_deterministic()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')

    # Dataset initialization
    try:
        dataset = MemoryEfficientGraphDataset(
            root_dir=args.graph_dir,
            graph_mapping_file=args.graph_mapping
        )
        train_data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {str(e)}")
        raise

    # Model initialization (single time)
    model = initialize_model(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        pool_ratio=args.pool_ratio,
        num_classes=args.num_classes,
        device=device
    )

    # After model initialization
    logger.info("Model configuration:")
    logger.info(f"Input dimension: {args.input_dim}")
    logger.info(f"Hidden dimension: {args.hidden_dim}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Number of heads: {args.num_heads}")
    logger.info(f"Pool ratio: {args.pool_ratio}")

    # Training configuration (single time)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = VirtualBatchMIWeightedQR(
        accumulation_steps=args.accumulation_steps,
        feature_dim=args.hidden_dim,
        weight_lr=args.weight_lr
    )

    # Initialize training variables
    use_custom_loss = False
    best_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):  # Changed from num_epochs
        logger.info(f'----------- EPOCH: {epoch} ----------')
        model.train()
        accumulated_loss = 0.0 
        total_loss = 0.0
        valid_batch_count = 0

        try:
            if epoch == args.custom_loss_epoch:  # Changed from SWITCH_TO_CUSTOM_LOSS_AT_EPOCH
                use_custom_loss = True
                logger.info('Switching to custom loss')
                best_loss = float('inf')

            for batch_idx, data in enumerate(train_data_loader):
                if batch_idx in args.skip_indices:  # Changed from skip_indices
                    logger.info(f"Skipping problematic graph {batch_idx}")
                    continue

                try:   
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()

                    data = data.to(device)
                    labels = data.y.float().view(-1, 2)  
                    outputs, penultimate = model(data)
                    outputs = outputs.view(-1, 2)

                    if use_custom_loss:
                        loss = loss_function(outputs, labels, penultimate)
                    else:
                        loss = nn.BCEWithLogitsLoss()(outputs, labels) / args.accumulation_steps
            
                    loss.backward()
                    valid_batch_count += 1
                    
                    total_loss += loss.item()
                    accumulated_loss += loss.item()

                    if (batch_idx + 1) % args.log_frequency == 0:  # Changed from log_frequency_steps
                        logger.info(f'Step {batch_idx + 1}: Model outputs: {torch.sigmoid(outputs.detach().cpu())}')

                    if valid_batch_count >= args.accumulation_steps or (batch_idx + 1) == len(train_data_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        bloss = accumulated_loss / valid_batch_count
                        logger.info(f'Step {batch_idx + 1}: Loss: {bloss*100}')
                        logger.info(f'Valid batches: {valid_batch_count}')
                        
                        accumulated_loss = 0.0 
                        valid_batch_count = 0

                except RuntimeError as e:
                    logger.error(f"OOM error at batch {batch_idx}")
                    loss_function.reset_storage()
                    raise

        finally:
            loss_function.cleanup()
            
        average_epoch_loss = total_loss / len(train_data_loader)
        logger.info(f'Epoch {epoch} average loss: {average_epoch_loss}')

        if average_epoch_loss < best_loss:
            best_loss = average_epoch_loss
            model_path = f"MI_QR_model_epoch_{epoch+1}_best.pth"
            model_save_path = os.path.join(args.model_save_dir, model_path)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Saved best model at epoch {epoch + 1}')
