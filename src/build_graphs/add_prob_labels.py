import pandas as pd
import numpy as np
import torch
import joblib
import logging
import os
import json
from datetime import datetime
from tqdm import tqdm
from torch_geometric.data import Data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process and label graph data')
    
    # Create argument groups
    io_args = parser.add_argument_group('Input/Output')
    data_args = parser.add_argument_group('Data Processing')
    system_args = parser.add_argument_group('System')
    
    # Input/Output arguments
    io_args.add_argument('--base-dir', required=True,
                      help='Base directory for all operations')
    io_args.add_argument('--output-dir', type=str,
                      help='Output directory (defaults to base-dir if not specified)')
    io_args.add_argument('--log-dir', type=str,
                      help='Directory for log files (defaults to base-dir/logs if not specified)')
    io_args.add_argument('--graph-dir', type=str,
                      help='Directory containing input graph files')
    io_args.add_argument('--labels-path', type=str,
                      help='Path to labels file (.pkl)')
    
    # Data processing arguments
    data_args.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for saving graphs')
    data_args.add_argument('--prefix', type=str, default='processed',
                        help='Prefix for output files')
    data_args.add_argument('--save-individually', action='store_true',
                        help='Save graphs individually instead of in batches')
    data_args.add_argument('--mrn-column', type=str, default='MRN',
                        help='Name of MRN column in labels DataFrame')
    data_args.add_argument('--label-column', type=str, default='prob_array',
                        help='Name of label column in labels DataFrame')
    
    # System arguments
    system_args.add_argument('--debug', action='store_true',
                          help='Enable debug logging')
    system_args.add_argument('--num-workers', type=int, default=1,
                          help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Set default output_dir if not specified
    if args.output_dir is None:
        args.output_dir = args.base_dir
        
    # Set default log_dir if not specified
    if args.log_dir is None:
        args.log_dir = os.path.join(args.base_dir, 'logs')
    
    # Validate arguments
    if not os.path.exists(args.base_dir):
        parser.error(f"Base directory does not exist: {args.base_dir}")
    
    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    return args

def save_graphs_in_batches(labeled_graphs, output_dir, prefix, timestamp, batch_size=100, logger=None):
    graphs_dir = os.path.join(output_dir, f'{prefix}_graphs_{timestamp}')
    os.makedirs(graphs_dir, exist_ok=True)
    mrn_to_file = {}
    
    mrns = list(labeled_graphs.keys())
    for i in range(0, len(mrns), batch_size):
        batch_mrns = mrns[i:i + batch_size]
        batch_graphs = {mrn: labeled_graphs[mrn] for mrn in batch_mrns}
        
        filename = f'graphs_batch_{i//batch_size}.pkl'
        filepath = os.path.join(graphs_dir, filename)
        
        try:
            joblib.dump(batch_graphs, filepath)
            for mrn in batch_mrns:
                mrn_to_file[mrn] = os.path.join(os.path.basename(graphs_dir), filename)
        except Exception as e:
            logger.error(f"Error saving batch {i//batch_size}: {str(e)}")
            
    return mrn_to_file

def debug_mrn_formats(graph_mrns, label_mrns, logger):
    """Debug MRN format differences"""
    logger.info("\nDEBUG MRN Formats:")
    logger.info(f"Number of graph MRNs: {len(graph_mrns)}")
    logger.info(f"Number of label MRNs: {len(label_mrns)}")
    
    # Sample some MRNs from each
    logger.info("\nSample graph MRNs (first 5):")
    for mrn in list(graph_mrns)[:5]:
        logger.info(f"  {mrn} (type: {type(mrn)}, length: {len(str(mrn))})")
    
    logger.info("\nSample label MRNs (first 5):")
    for mrn in list(label_mrns)[:5]:
        logger.info(f"  {mrn} (type: {type(mrn)}, length: {len(str(mrn))})")
    
    # Check for any MRNs that appear in both sets
    common_mrns = set(graph_mrns) & set(label_mrns)
    logger.info(f"\nNumber of matching MRNs: {len(common_mrns)}")
    if common_mrns:
        logger.info("Sample matching MRNs (first 5):")
        for mrn in list(common_mrns)[:5]:
            logger.info(f"  {mrn}")

def convert_label_to_tensor(label):
    """
    Convert label to PyTorch tensor regardless of input type.
    
    Parameters:
    -----------
    label : np.ndarray, list, or scalar
        Label to convert
        
    Returns:
    --------
    torch.Tensor
        Label as a PyTorch tensor
    """
    if isinstance(label, torch.Tensor):
        return label
    elif isinstance(label, (np.ndarray, list)):
        return torch.FloatTensor(label)
    else:
        return torch.tensor(label, dtype=torch.float)


def process_labels(graphs, label_df, mrn_column='MRN', label_column='prob_array', 
                  mode='add', desc="Processing labels", logger=None):
    """
    Process labels for graphs - either adding new or reassigning existing.
    
    Parameters:
    -----------
    graphs : dict
        Dictionary of MRN to PyG Data objects
    label_df : pd.DataFrame
        DataFrame containing MRN and label columns
    mrn_column : str
        Name of the MRN column in label_df
    label_column : str
        Name of the label column in label_df
    mode : str
        Either 'add' or 'reassign'
    desc : str
        Description for progress bar
        
    Returns:
    --------
    dict, dict
        Processed graphs dictionary and statistics
    """
    # Validate inputs
    if not isinstance(graphs, dict):
        raise ValueError("graphs must be a dictionary")
    if not isinstance(label_df, pd.DataFrame):
        raise ValueError("label_df must be a pandas DataFrame")
    if mrn_column not in label_df.columns:
        raise ValueError(f"MRN column '{mrn_column}' not found in DataFrame")
    if label_column not in label_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")
        
    # Create MRN to label mapping
    label_dict = dict(zip(label_df[mrn_column], label_df[label_column]))
    
    processed_graphs = {}
    stats = {
        'total_input_graphs': len(graphs),
        'total_input_labels': len(label_df),
        f'graphs_{mode}d': 0,
        'graphs_missing_labels': 0,
        'label_dimension': None,
        'included_mrns': [],
        'excluded_mrns': []
    }
    
    # Get label dimension from first available label
    first_label = label_df[label_column].iloc[0]
    stats['label_dimension'] = (len(first_label) 
                              if isinstance(first_label, (np.ndarray, list)) 
                              else 1)
    
    # Process graphs
    for mrn, graph in tqdm(graphs.items(), desc=desc):
        if mrn in label_dict:
            try:
                # Convert label to tensor
                label = convert_label_to_tensor(label_dict[mrn])
                
                # Validate label dimension
                label_dim = label.numel()
                if stats['label_dimension'] != label_dim:
                    logger.warning(
                        f"Inconsistent label dimension for MRN {mrn}: "
                        f"expected {stats['label_dimension']}, got {label_dim}"
                    )
                
                # Assign label to graph
                graph.y = label
                processed_graphs[mrn] = graph
                stats[f'graphs_{mode}d'] += 1
                stats['included_mrns'].append(mrn)
                
            except Exception as e:
                logger.error(f"Error processing label for MRN {mrn}: {str(e)}")
                stats['graphs_missing_labels'] += 1
                stats['excluded_mrns'].append(mrn)
        else:
            stats['graphs_missing_labels'] += 1
            stats['excluded_mrns'].append(mrn)
    
    # Sort MRN lists for consistency
    stats['included_mrns'].sort()
    stats['excluded_mrns'].sort()
    
    # Add summary statistics
    stats['final_dataset_size'] = len(processed_graphs)
    
    return processed_graphs, stats

def add_labels_to_graphs(pyg_graphs_dict, label_df, mrn_column='MRN', label_column='prob_array', logger=None):
    """
    Add labels to PyG Data objects in the graph dictionary.
    
    Parameters:
    -----------
    pyg_graphs_dict : dict
        Dictionary mapping MRN to PyG Data objects
    label_df : pd.DataFrame
        DataFrame containing MRN and label columns
    mrn_column : str
        Name of the MRN column in label_df
    label_column : str
        Name of the label column in label_df
        
    Returns:
    --------
    dict, dict
        Labeled graphs dictionary and statistics
    """
    return process_labels(
        pyg_graphs_dict, 
        label_df, 
        mrn_column=mrn_column,
        label_column=label_column,
        mode='add',
        desc="Adding labels to graphs",
        logger=logger  
    )

def reassign_labels(labeled_graphs, new_label_df, mrn_column='MRN', label_column='prob_array', logger=None):
    """
    Reassign labels to existing labeled graphs.
    
    Parameters:
    -----------
    labeled_graphs : dict
        Dictionary of MRN to PyG Data objects
    new_label_df : pd.DataFrame
        DataFrame with MRN and new label columns
    mrn_column : str
        Name of the MRN column in new_label_df
    label_column : str
        Name of the label column in new_label_df
        
    Returns:
    --------
    dict, dict
        Updated graphs dictionary and statistics
    """
    return process_labels(
        labeled_graphs, 
        new_label_df, 
        mrn_column=mrn_column,
        label_column=label_column,
        mode='relabel',
        desc="Reassigning labels",
        logger=logger  # Pass the logger parameter through
    )


def save_graphs_individually(labeled_graphs, output_dir, prefix, timestamp, logger=None):
    """
    Save each graph as a separate PyTorch file.
    
    Parameters:
    -----------
    labeled_graphs : dict
        Dictionary of MRN to PyG Data objects
    output_dir : str
        Directory to save the graph files
    prefix : str
        Prefix for the graph files (e.g., 'normalized' or 'non_normalized')
    
    Returns:
    --------
    dict
        Dictionary mapping MRNs to their corresponding filenames
    """
    # Create a subdirectory for the graphs
    graphs_dir = os.path.join(output_dir, f'{prefix}_graphs_{timestamp}')
    os.makedirs(graphs_dir, exist_ok=True)
    
    mrn_to_file = {}
    
    for mrn, graph in tqdm(labeled_graphs.items(), desc=f"Saving {prefix} graphs"):
        # Create filename for this graph
        filename = f'graph_{mrn}.pth'
        filepath = os.path.join(graphs_dir, filename)
        
        # Save individual graph
        torch.save(graph, filepath)
        
        # Store mapping
        mrn_to_file[mrn] = os.path.join(os.path.basename(graphs_dir), filename)
    
    # Save the MRN to filename mapping
    mapping_file = os.path.join(output_dir, f'{prefix}_mrn_to_file_{timestamp}.json')
    with open(mapping_file, 'w') as f:
        json.dump(mrn_to_file, f, indent=4)
        
    return mrn_to_file

def main():
    logger = None
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set up logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
        
        # Reset handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(args.log_dir, f'process_graphs_{timestamp}.log')
        
        # Create handlers
        file_handler = logging.FileHandler(log_file_path)
        console_handler = logging.StreamHandler()
        
        # Set format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info("Starting graph processing...")
        logger.info(f"Arguments: {vars(args)}")
        
        # Load existing graphs
        logger.info("Loading graphs...")
        input_graphs = {}
        failed_loads = 0
        
        if args.graph_dir:
            if not os.path.exists(args.graph_dir):
                raise ValueError(f"Graph directory does not exist: {args.graph_dir}")
            
            total_files = len([f for f in os.listdir(args.graph_dir) if f.endswith('.pth')])
            logger.info(f"Found {total_files} .pth files in graph directory")
                
            for file in tqdm(os.listdir(args.graph_dir), desc="Loading graphs"):
                if file.endswith('.pth'):
                    try:
                        filepath = os.path.join(args.graph_dir, file)
                        mrn = file.split('_')[1].split('.')[0]
                        graph = torch.load(filepath)
                        
                        # Validate graph structure
                        if not isinstance(graph, Data):
                            logger.warning(f"Skipping {file}: not a PyG Data object")
                            failed_loads += 1
                            continue
                            
                        # Additional graph validation
                        if not hasattr(graph, 'x') or graph.x is None:
                            logger.warning(f"Skipping {file}: missing node features")
                            failed_loads += 1
                            continue
                            
                        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
                            logger.warning(f"Skipping {file}: missing edges")
                            failed_loads += 1
                            continue
                            
                        input_graphs[mrn] = graph
                    except Exception as e:
                        logger.error(f"Error loading {file}: {str(e)}")
                        failed_loads += 1
                        
            logger.info(f"Loaded {len(input_graphs)} graphs successfully")
            if failed_loads > 0:
                logger.warning(f"Failed to load {failed_loads} graphs")
        else:
            logger.warning("No graph directory specified, starting with empty graph set")
        
        # Load and process labels if specified
        processed_graphs = None
        stats = None
        
        if args.labels_path:
            if not os.path.exists(args.labels_path):
                raise ValueError(f"Labels file does not exist: {args.labels_path}")
                
            logger.info("Loading labels...")
            try:
                label_df = joblib.load(args.labels_path)
                if label_df.empty:
                    raise ValueError("Label DataFrame is empty")
                    
                logger.info(f"Loaded labels for {len(label_df)} patients")
                
                # Validate label DataFrame
                if args.mrn_column not in label_df.columns:
                    raise ValueError(f"MRN column '{args.mrn_column}' not found in labels")
                if args.label_column not in label_df.columns:
                    raise ValueError(f"Label column '{args.label_column}' not found in labels")
                
                # Check for NaN values
                if label_df[args.label_column].isna().any():
                    raise ValueError("Label column contains NaN values")
                
                # Print DataFrame info
                logger.info("\nLabel DataFrame columns:")
                logger.info(label_df.columns.tolist())
                logger.info(f"\nLabel DataFrame shape: {label_df.shape}")
                
                # Debug MRN formats if needed
                if args.debug:
                    debug_mrn_formats(
                        set(input_graphs.keys()), 
                        set(label_df[args.mrn_column]),
                        logger
                    )
                
                # Process graphs according to whether we're adding or reassigning labels
                if input_graphs:
                    logger.info("Reassigning labels to existing graphs...")
                    processed_graphs, stats = reassign_labels(
                        input_graphs, 
                        label_df,
                        mrn_column=args.mrn_column,
                        label_column=args.label_column,
                        logger=logger  # Add logger parameter
                    )
                else:
                    logger.info("Adding labels to new graphs...")
                    processed_graphs, stats = add_labels_to_graphs(
                        input_graphs, 
                        label_df,
                        mrn_column=args.mrn_column,
                        label_column=args.label_column,
                        logger=logger  # Add logger parameter
                    )
            except Exception as e:
                raise ValueError(f"Error processing labels: {str(e)}")
        else:
            processed_graphs = input_graphs
            stats = {
                'total_input_graphs': len(input_graphs),
                'graphs_processed': len(input_graphs),
                'included_mrns': sorted(list(input_graphs.keys()))
            }
        
        # Save processed graphs
        if processed_graphs:
            try:
                if args.save_individually:
                    logger.info("Saving graphs individually...")
                    mrn_file_mapping = save_graphs_individually(
                        processed_graphs, 
                        args.output_dir, 
                        args.prefix,
                        timestamp,
                        logger
                    )
                else:
                    logger.info(f"Saving graphs in batches of {args.batch_size}...")
                    mrn_file_mapping = save_graphs_in_batches(
                        processed_graphs, 
                        args.output_dir, 
                        args.prefix,
                        timestamp,
                        args.batch_size,
                        logger
                    )
                
                # Update stats with file mapping
                stats['graph_files'] = mrn_file_mapping
                
                # Save MRN lists
                included_mrns_file = os.path.join(args.output_dir, f'included_mrns_{timestamp}.txt')
                excluded_mrns_file = os.path.join(args.output_dir, f'excluded_mrns_{timestamp}.txt')
                
                if 'included_mrns' in stats:
                    with open(included_mrns_file, 'w') as f:
                        for mrn in sorted(stats['included_mrns']):
                            f.write(f"{mrn}\n")
                
                if 'excluded_mrns' in stats:
                    with open(excluded_mrns_file, 'w') as f:
                        for mrn in sorted(stats['excluded_mrns']):
                            f.write(f"{mrn}\n")
                
                # Save stats
                stats_output = os.path.join(args.output_dir, f'{args.prefix}_stats_{timestamp}.json')
                with open(stats_output, 'w') as f:
                    json.dump(stats, f, indent=4)
                
                # Log final summary
                logger.info("\nProcessing Summary:")
                logger.info(f"Total input graphs: {stats.get('total_input_graphs', 0)}")
                logger.info(f"Graphs processed: {stats.get('graphs_processed', 0)}")
                if 'graphs_missing_labels' in stats:
                    logger.info(f"Graphs missing labels: {stats['graphs_missing_labels']}")
                
                # Log output files
                logger.info("\nSaved files:")
                logger.info(f"Graphs directory: {os.path.join(args.output_dir, f'{args.prefix}_graphs_{timestamp}')}")
                logger.info(f"Stats file: {stats_output}")
                logger.info(f"Included MRNs list: {included_mrns_file}")
                if 'excluded_mrns' in stats:
                    logger.info(f"Excluded MRNs list: {excluded_mrns_file}")
                    
            except Exception as e:
                raise ValueError(f"Error saving processed graphs: {str(e)}")
        else:
            logger.warning("No graphs to save!")
            
    except Exception as e:
        if logger:
            logger.error(f"Error in main process: {str(e)}", exc_info=True)
        raise
    
    finally:
        if logger:
            logger.info("Process completed")
            # Clean up handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

if __name__ == "__main__":
    main()

