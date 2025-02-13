import pandas as pd
import logging
import joblib
import torch
import os
import psutil
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from datetime import datetime
from tqdm import tqdm
import argparse
from Model import Date2VecConvert

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Build and process graph data with customizable paths')
    
    # Required arguments
    parser.add_argument('--log-dir', required=True,
                      help='Directory for saving log files')
    parser.add_argument('--graph-tuples', required=True,
                      help='Path to graph_ready_tuples_sem.bz2')
    parser.add_argument('--embeddings', required=True,
                      help='Path to graph_ready_tuples_embeddings.bz2')
    parser.add_argument('--time-embeddings', required=True,
                      help='Path to time_embeddings dictionary')
    parser.add_argument('--output-dir', required=True,
                      help='Directory for saving output files')
    
    # Optional arguments
    parser.add_argument('--d2v-model', 
                      help='Path to Date2Vec model')
    parser.add_argument('--run-name', default='graph_construction',
                      help='Name for this run (used in logging)')
    parser.add_argument('--semantic', action='store_true',
                      help='Whether to include semantic embeddings')
    
    return parser.parse_args()

def setup_logging(log_dir, run_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{run_name}.log')
    
    file_handler = logging.FileHandler(log_file_path, mode='a')
    tqdm_handler = TqdmLoggingHandler()
    
    file_handler.setLevel(logging.INFO)
    tqdm_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    tqdm_handler.setFormatter(formatter)
    
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(tqdm_handler)
    
    return logger

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert bytes to GB

def log_memory(logger, step):
    """Log current memory usage"""
    memory_gb = get_memory_usage()
    logger.info(f"Memory usage after {step}: {memory_gb:.2f} GB")

def build_graph(tuples_df, time_embeddings_dict, node_embeddings_dict, semantic=True):
    """Build directed graphs from tuples dataframe with node and time embeddings."""
    # Input validation
    required_cols = ['MRN']
    required_cols.append('CombinedTupleListSem' if semantic else 'CombinedTuples')
    
    if not all(col in tuples_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
    graphs = {}
    missing_embeddings = {
        'node': set(),
        'time': set()
    }
    if semantic:
        missing_embeddings['semantic'] = set()
    
    for row in tuples_df.itertuples():
        G_i = nx.DiGraph()
        mrn = row.MRN
        
        tuple_attr = 'CombinedTupleListSem' if semantic else 'CombinedTuples'
        graph_tuples = sorted(getattr(row, tuple_attr), key=lambda x: x[1])
        
        valid_nodes = []
        
        for idx, graph_node in enumerate(graph_tuples):
            node_id = graph_node[0]
            timestamp = graph_node[1]
            dt_timestamp = pd.to_datetime(timestamp)
            
            node_emb = node_embeddings_dict.get(node_id)
            time_emb = time_embeddings_dict.get(dt_timestamp)
            
            if semantic:
                sem_id = graph_node[2]
                node_emb_sem = node_embeddings_dict.get(sem_id)
                if node_emb_sem is None:
                    missing_embeddings['semantic'].add(sem_id)
            
            if node_emb is None:
                missing_embeddings['node'].add(node_id)
            if time_emb is None:
                missing_embeddings['time'].add(dt_timestamp)
            
            embeddings_available = False
            if semantic:
                if all([node_emb is not None, node_emb_sem is not None, time_emb is not None]):
                    node_feature = np.concatenate([node_emb, node_emb_sem, time_emb], axis=0)
                    embeddings_available = True
            else:
                if all([node_emb is not None, time_emb is not None]):
                    node_feature = np.concatenate([node_emb, time_emb], axis=0)
                    embeddings_available = True
            
            if embeddings_available:
                G_i.add_node(idx, x=node_feature)
                valid_nodes.append(idx)
                
                for prev_idx in valid_nodes[:-1]:
                    G_i.add_edge(prev_idx, idx, timestamp=dt_timestamp)
        
        if G_i.number_of_nodes() > 0:
            graphs[mrn] = G_i
            
    return graphs, missing_embeddings

def build_graph_tqdm(tuples_df, time_embeddings_dict, node_embeddings_dict, semantic=True, output_dir=None):
    """Build directed graphs with progress bar."""
    required_cols = ['MRN']
    required_cols.append('CombinedTupleListSem' if semantic else 'CombinedTuples')
    
    if not all(col in tuples_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
    graphs = {}
    missing_embeddings = {
        'node': set(),
        'time': set()
    }
    if semantic:
        missing_embeddings['semantic'] = set()
    
    mrn_missing_embeddings = {
        'node': {},
        'time': {}
    }
    if semantic:
        mrn_missing_embeddings['semantic'] = {}
    
    for row in tqdm(tuples_df.itertuples(), total=len(tuples_df), desc="Building patient graphs"):
        G_i = nx.DiGraph()
        mrn = row.MRN
        
        tuple_attr = 'CombinedTupleListSem' if semantic else 'CombinedTuples'
        graph_tuples = sorted(getattr(row, tuple_attr), key=lambda x: x[1])
        
        valid_nodes = []
        mrn_missing = {
            'node': set(),
            'time': set()
        }
        if semantic:
            mrn_missing['semantic'] = set()
        
        for idx, graph_node in enumerate(graph_tuples):
            node_id = graph_node[0]
            timestamp = graph_node[1]
            dt_timestamp = pd.to_datetime(timestamp)
            
            node_emb = node_embeddings_dict.get(node_id)
            time_emb = time_embeddings_dict.get(dt_timestamp)
            
            if semantic:
                sem_id = graph_node[2]
                node_emb_sem = node_embeddings_dict.get(sem_id)
                if node_emb_sem is None:
                    missing_embeddings['semantic'].add(sem_id)
                    mrn_missing['semantic'].add(sem_id)
            
            if node_emb is None:
                missing_embeddings['node'].add(node_id)
                mrn_missing['node'].add(node_id)
            if time_emb is None:
                missing_embeddings['time'].add(dt_timestamp)
                mrn_missing['time'].add(dt_timestamp)
            
            embeddings_available = False
            if semantic:
                if all([node_emb is not None, node_emb_sem is not None, time_emb is not None]):
                    node_feature = np.concatenate([node_emb, node_emb_sem, time_emb], axis=0)
                    embeddings_available = True
            else:
                if all([node_emb is not None, time_emb is not None]):
                    node_feature = np.concatenate([node_emb, time_emb], axis=0)
                    embeddings_available = True
            
            if embeddings_available:
                G_i.add_node(idx, x=node_feature)
                valid_nodes.append(idx)
                
                for prev_idx in valid_nodes[:-1]:
                    G_i.add_edge(prev_idx, idx, timestamp=dt_timestamp)
        
        for emb_type in mrn_missing:
            if mrn_missing[emb_type]:
                mrn_missing_embeddings[emb_type][mrn] = mrn_missing[emb_type]
        
        if G_i.number_of_nodes() > 0:
            graphs[mrn] = G_i
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_dir and any(missing_embeddings.values()):
        for emb_type, missing in missing_embeddings.items():
            if missing:
                missing_file = os.path.join(output_dir, f'missing_{emb_type}_embeddings_{timestamp}.txt')
                mrn_missing_file = os.path.join(output_dir, f'mrn_missing_{emb_type}_embeddings_{timestamp}.txt')
                
                with open(missing_file, 'w') as f:
                    f.write(f"All missing {emb_type} embeddings:\n")
                    for value in sorted(missing):
                        f.write(f"{value}\n")
                
                with open(mrn_missing_file, 'w') as f:
                    f.write(f"Missing {emb_type} embeddings by MRN:\n")
                    for mrn, missing_vals in mrn_missing_embeddings[emb_type].items():
                        f.write(f"\nMRN: {mrn}\n")
                        for value in sorted(missing_vals):
                            f.write(f"    {value}\n")
    
    return graphs, missing_embeddings

def convert_to_pyg_data(graphs_dict):
    """Convert dictionary of networkx graphs to PyG Data objects."""
    pyg_graphs = {}
    
    for mrn, nx_graph in graphs_dict.items():
        node_features = torch.FloatTensor([nx_graph.nodes[i]['x'] for i in range(len(nx_graph))])
        edge_index = torch.LongTensor([[u, v] for u, v in nx_graph.edges()]).t()
        edge_attr = torch.FloatTensor([
            [nx_graph.edges[u, v]['timestamp'].timestamp()] 
            for u, v in nx_graph.edges()
        ])
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        pyg_graphs[mrn] = data
    
    return pyg_graphs

def normalize_pyg_graphs_memory_efficient(pyg_graphs_dict):
    """Normalize PyG graphs efficiently."""
    feature_sum = None
    total_nodes = 0
    
    for data in pyg_graphs_dict.values():
        if feature_sum is None:
            feature_sum = torch.zeros(data.x.shape[1])
        feature_sum += torch.sum(data.x, dim=0)
        total_nodes += data.x.shape[0]
    
    feature_mean = feature_sum / total_nodes

    squared_diff_sum = torch.zeros_like(feature_sum)
    for data in pyg_graphs_dict.values():
        diff = data.x - feature_mean
        squared_diff_sum += torch.sum(diff * diff, dim=0)
    
    feature_std = torch.sqrt(squared_diff_sum / total_nodes)
    feature_std = torch.where(feature_std == 0, 1e-8, feature_std)

    normalized_graphs = {}
    for mrn, data in pyg_graphs_dict.items():
        normalized_x = (data.x - feature_mean) / feature_std
        data.x = normalized_x
        normalized_graphs[mrn] = data
    
    return normalized_graphs, {
        'mean': feature_mean,
        'std': feature_std
    }

def update_time_embeddings(missing_times, existing_time_embeddings_dict, d2v_model_path, output_dir):
    """Update time embeddings dictionary with missing timestamps."""
    logger = logging.getLogger(__name__)
    logger.info(f"Updating time embeddings for {len(missing_times)} missing timestamps")
    
    missing_df = pd.DataFrame({'universal_time': list(missing_times)})
    missing_df['datetime'] = pd.to_datetime(missing_df['universal_time'])
    
    def convert_datetime_to_tensor(x):
        return torch.tensor([x.hour, x.minute, x.second, x.year, x.month, x.day], dtype=torch.float)
    
    missing_df['tensor_column'] = missing_df['datetime'].apply(convert_datetime_to_tensor)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d2v = Date2VecConvert(model_path=d2v_model_path)
    
    def make_embeddings_tt(row):
        time = row['tensor_column']
        time = time.to(device)
        embedding = d2v(time)
        return embedding.cpu().data
    
    logger.info("Generating new embeddings...")
    tqdm.pandas()
    missing_df['t_embeddings'] = missing_df.progress_apply(make_embeddings_tt, axis=1)
    
    new_embeddings = dict(zip(missing_df['datetime'], missing_df['t_embeddings']))
    updated_dict = existing_time_embeddings_dict.copy()
    updated_dict.update(new_embeddings)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'time_embeddings_dict_updated_{timestamp}.pkl')
    joblib.dump(updated_dict, output_path)
    
    logger.info(f"Updated dictionary saved to: {output_path}")
    logger.info(f"Original dictionary size: {len(existing_time_embeddings_dict)}")
    logger.info(f"New dictionary size: {len(updated_dict)}")
    
    return updated_dict

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.run_name)
    logger.info("Starting graph construction process...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Load data
        logger.info("Loading input files...")
        start_time = datetime.now()
        
        graph_tuples = joblib.load(args.graph_tuples)
        log_memory(logger, "loading graph tuples")
        
        graph_tuple_embeddings = joblib.load(args.embeddings)
        log_memory(logger, "loading embeddings")
        
        time_embeddings = joblib.load(args.time_embeddings)
        log_memory(logger, "loading time embeddings")
        
        # Build NetworkX graphs
        logger.info("Building NetworkX graphs...")
        graphs, missing_embeddings = build_graph_tqdm(
            tuples_df=graph_tuples,
            time_embeddings_dict=time_embeddings,
            node_embeddings_dict=graph_tuple_embeddings,
            semantic=args.semantic,
            output_dir=args.output_dir
        )
        logger.info(f'Missing embeddings: {missing_embeddings}')
        log_memory(logger, "building NetworkX graphs")
        logger.info(f"Built graphs for {len(graphs)} patients")
        
        # Handle missing time embeddings if d2v_model is provided
        if args.d2v_model and any(missing_embeddings['time']):
            logger.info("Detected missing time embeddings. Attempting to update...")
            try:
                time_embeddings = update_time_embeddings(
                    missing_times=missing_embeddings['time'],
                    existing_time_embeddings_dict=time_embeddings,
                    d2v_model_path=args.d2v_model,
                    output_dir=args.output_dir
                )
                
                # Rebuild graphs with updated time embeddings
                logger.info("Rebuilding graphs with updated time embeddings...")
                graphs, missing_embeddings = build_graph_tqdm(
                    tuples_df=graph_tuples,
                    time_embeddings_dict=time_embeddings,
                    node_embeddings_dict=graph_tuple_embeddings,
                    semantic=args.semantic,
                    output_dir=args.output_dir
                )
                log_memory(logger, "rebuilding NetworkX graphs with updated time embeddings")
                logger.info(f"Rebuilt graphs for {len(graphs)} patients")
                
                if any(missing_embeddings['time']):
                    logger.error("Still found missing time embeddings after update!")
                    raise ValueError("Failed to resolve all missing time embeddings")
                    
            except Exception as e:
                logger.error(f"Error updating time embeddings: {str(e)}")
                raise
        
        # Convert to PyG Data objects
        logger.info("Converting to PyG Data objects...")
        try:
            pyg_graphs = convert_to_pyg_data(graphs)
            log_memory(logger, "converting to PyG")
            
            # Clear NetworkX graphs to save memory
            del graphs
            log_memory(logger, "after deleting NetworkX graphs")
            
            # Save non-normalized graphs
            unscaled_graphs_file = os.path.join(args.output_dir, f'non_normalized_graphs_{timestamp}.pkl')
            joblib.dump(pyg_graphs, unscaled_graphs_file)
            logger.info(f"Saved non-normalized graphs to {unscaled_graphs_file}")
            
        except Exception as e:
            logger.error(f"Error converting to PyG: {str(e)}")
            raise
            
        # Normalize features
        logger.info("Normalizing features...")
        try:
            normalized_graphs, normalization_stats = normalize_pyg_graphs_memory_efficient(pyg_graphs)
            log_memory(logger, "normalizing graphs")
            
            # Clear unnormalized graphs to save memory
            del pyg_graphs
            log_memory(logger, "after deleting unnormalized graphs")
            
            # Save normalization stats and normalized graphs
            norm_stats_file = os.path.join(args.output_dir, f'normalization_stats_{timestamp}.pkl')
            joblib.dump(normalization_stats, norm_stats_file)
            logger.info(f"Saved normalization stats to {norm_stats_file}")
            
            normalized_graphs_file = os.path.join(args.output_dir, f'normalized_graphs_{timestamp}.pkl')
            joblib.dump(normalized_graphs, normalized_graphs_file)
            logger.info(f"Saved normalized graphs to {normalized_graphs_file}")
            
            # Print summary statistics
            total_nodes = sum(data.num_nodes for data in normalized_graphs.values())
            total_edges = sum(data.num_edges for data in normalized_graphs.values())
            feature_dim = next(iter(normalized_graphs.values())).num_node_features
            
            logger.info("\nDataset Statistics:")
            logger.info(f"Total number of patients: {len(normalized_graphs)}")
            logger.info(f"Total nodes across all graphs: {total_nodes}")
            logger.info(f"Total edges across all graphs: {total_edges}")
            logger.info(f"Average nodes per patient: {total_nodes / len(normalized_graphs):.2f}")
            logger.info(f"Average edges per patient: {total_edges / len(normalized_graphs):.2f}")
            logger.info(f"Feature dimension: {feature_dim}")
            
            # Calculate total processing time
            end_time = datetime.now()
            processing_time = end_time - start_time
            logger.info(f"\nTotal processing time: {processing_time}")
            
        except Exception as e:
            logger.error(f"Error in normalization and saving: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}")
        raise
        
    finally:
        log_memory(logger, "completion")

if __name__ == "__main__":
    main()
