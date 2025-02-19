import os
import sys
import logging
import random
import argparse
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import dill as pickle
from tqdm import tqdm
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, LabelModel

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Type aliases
RuleSet = Set[str]
RuleMetadata = Tuple[str, pd.DataFrame, Set, int]

@dataclass
class RuleGenerationConfig:
    """Configuration for rule generation parameters."""
    label_to_select: int
    percentage_positive: float
    percentage_negative: float
    rules_dimension_positive: int
    rules_dimension_negative: int
    min_length: int = 25
    seed: Optional[int] = None

def setup_logger(log_dir: str) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'processing_{datetime.now():%Y%m%d_%H%M%S}.log'))
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_arguments():
    """Parse command line arguments for rule generation"""
    parser = argparse.ArgumentParser(description='Generate and manage labeling rules for EHR classification')
    
    # Required arguments
    parser.add_argument('--nodes-path', required=True,
                      help='Path to processed nodes file')
    parser.add_argument('--source-target-path', required=True,
                      help='Path to source-target mapping file')
    parser.add_argument('--labels-dict-path', required=True,
                      help='Path to labels dictionary file')
    parser.add_argument('--output-dir', required=True,
                      help='Directory for saving output files')
    parser.add_argument('--log-dir', required=True,
                      help='Directory for saving log files')
    
    # Optional arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--class-type', choices=['positive', 'negative'], default='positive',
                      help='Type of rules to generate')
    parser.add_argument('--quantity', type=int, default=100,
                      help='Number of rules to generate')
    parser.add_argument('--percentage-positive', type=float, default=0.4,
                      help='Percentage threshold for positive rules')
    parser.add_argument('--percentage-negative', type=float, default=0.65,
                      help='Percentage threshold for negative rules')
    parser.add_argument('--rules-dimension', type=int, default=4,
                      help='Dimension of rules to generate')
    parser.add_argument('--min-length', type=int, default=25,
                      help='Minimum length for generated rules')
    parser.add_argument('--job-index', type=int,
                      help='LSB job index for batch processing')
    
    return parser.parse_args()

class GraphBuilder:
    """Handles creation and manipulation of directed graphs."""
    
    @staticmethod
    def create_from_sorted_list(sorted_list: List[Tuple]) -> nx.DiGraph:
        """Create a directed graph from a sorted list of tuples."""
        G = nx.DiGraph()
        
        for i, (node_id, *_) in enumerate(sorted_list):
            G.add_node(node_id)
            # Add edges to all subsequent nodes
            for next_node_id, *_ in sorted_list[i+1:]:
                G.add_edge(node_id, next_node_id)
                
        return G

class LabelingFunctionGenerator:
    """Generates labeling functions for classification."""
    
    @staticmethod
    def create_functions(rule_ids: List[RuleSet], 
                        label_id: int, 
                        batch: int, 
                        label_key: Dict[int, int] = {0: 0, 1: 1}) -> Tuple[List, str]:
        """Create labeling functions from rule IDs."""
        functions = []
        
        def create_labeling_function(people: Set[str], func_name: str, label: int):
            @labeling_function()
            def new_func(x):
                return label if x.Source in people else -1
            new_func.__name__ = func_name
            return new_func
        
        for i, rule_set in enumerate(rule_ids):
            i_name = f"{i + np.random.randint(0,1000000)}{random.choice('ABCDEFGHIJKL')}"
            label_id_key = label_key[label_id]
            func_name = f"new_func_{i_name}_{label_id_key}_{batch}"
            
            new_func = create_labeling_function(rule_set, func_name, label_id_key)
            functions.append(new_func)
            
        return functions, func_name

class RuleGenerator:
    """Handles the generation of classification rules."""
    
    def __init__(self, nodes_df: pd.DataFrame, source_target: pd.DataFrame, dict_set: Dict, logger: logging.Logger):
        self.nodes_df = nodes_df
        self.source_target = source_target
        self.dict_set = dict_set
        self.logger = logger
        
    def get_rules(self, 
                  source_id: str,
                  label: int,
                  percentage: float,
                  rules_dimension: int) -> Tuple[Optional[pd.DataFrame], Optional[Set], Optional[Set]]:
        """Generate rules for a given source ID."""
        G = GraphBuilder.create_from_sorted_list(
            self.nodes_df[self.nodes_df['Source'] == source_id]['SortedNodes'].values[0]
        )
        
        max_iterations = 500
        for iteration in range(max_iterations):
            try:
                node_choice = random.choice(list(G.nodes()))
                nodes_list = random.sample(list(G.neighbors(node_choice)), k=10)
                
                # Get candidate nodes
                candidate_nodes = (
                    self.source_target[self.source_target['TargetsList'].isin(nodes_list)]
                    .groupby('TargetsList')['Source']
                    .aggregate(set)
                    .reset_index()
                )
                
                # Calculate overlap percentage
                candidate_nodes['overlap_percentage'] = candidate_nodes['Source'].apply(
                    lambda x: len(x.intersection(self.dict_set[label]))/len(x)
                )
                
                rules_select = candidate_nodes[candidate_nodes['overlap_percentage'] > percentage]
                
                if not rules_select.empty and len(rules_select) >= rules_dimension:
                    selected_rules = rules_select.sample(rules_dimension)
                    union_set = set.union(*selected_rules['Source'].to_list())
                    intersection_set = set.intersection(*selected_rules['Source'].to_list())
                    return selected_rules.copy(), union_set, intersection_set
                
            except Exception as e:
                self.logger.warning(f"Iteration {iteration} failed: {str(e)}")
                
        self.logger.info('No rules generated')
        return None, None, None

class RuleStorage:
    """Handles saving and loading of rules and metadata."""
    
    @staticmethod
    def write_rules(rules: List, filename: str):
        """Save rules to file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(rules, f)
    
    @staticmethod
    def read_rules(filename: str) -> List:
        """Load rules from file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def write_metadata(metadata: List[RuleMetadata], filename: str):
        """Save rule metadata to file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(metadata, f)
    
    @staticmethod
    def read_metadata(filename: str) -> List[RuleMetadata]:
        """Load rule metadata from file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logger(args.log_dir)
    logger.info(f"Starting rule generation with arguments: {args}")
    
    # Get job index if provided, otherwise use 0
    job_index = args.job_index if args.job_index is not None else 0
    
    # Adjust seed based on job index
    seed = args.seed + job_index
    random.seed(seed)
    np.random.seed(seed)
    
    # Set up paths
    label_to_select = 1 if args.class_type == 'positive' else 0
    rules_path = os.path.join(args.output_dir, f'generated_rules_{label_to_select}')
    metadata_path = os.path.join(args.output_dir, f'generated_rules_{label_to_select}_metadata')
    
    # Load data
    logger.info("Loading input data...")
    try:
        nodes = pd.read_pickle(args.nodes_path)
        source_target = pd.read_pickle(args.source_target_path)
        labels_dict = pd.read_pickle(args.labels_dict_path)
    except Exception as e:
        logger.error(f"Error loading input files: {str(e)}")
        sys.exit(1)
    
    # Process nodes
    nodes['Label'] = nodes['Source'].map(labels_dict)
    nodes['SortedNodes'] = nodes['ValidNodes'].apply(lambda x: sorted(x, key=lambda x: x[1]))
    
    # Set up configuration
    config = RuleGenerationConfig(
        label_to_select=label_to_select,
        percentage_positive=args.percentage_positive,
        percentage_negative=args.percentage_negative,
        rules_dimension_positive=args.rules_dimension,
        rules_dimension_negative=args.rules_dimension,
        quantity=args.quantity,
        seed=seed
    )
    
    # Initialize rule generator
    dict_set = dict(zip(nodes.groupby('Label')['Source'].agg(set).index, 
                       nodes.groupby('Label')['Source'].agg(set).values))
    generator = RuleGenerator(nodes, source_target, dict_set, logger)
    
    # Generate rules
    logger.info("Generating rules...")
    sources_list = nodes[nodes['Label'] == 0]['Source'].to_list()
    generated_rules = []
    rules_metadata = []
    
    with tqdm(total=args.quantity) as pbar:
        while len(generated_rules) < args.quantity:
            source_id = random.choice(sources_list)
            rules = generator.get_rules(
                source_id=source_id,
                label=label_to_select,
                percentage=args.percentage_positive if label_to_select == 1 else args.percentage_negative,
                rules_dimension=args.rules_dimension
            )
            
            if rules[0] is not None and len(rules[1]) >= args.min_length:
                function, func_name = LabelingFunctionGenerator.create_functions(
                    [rules[1]], label_to_select, seed
                )
                rules_metadata.append((func_name, rules[0], rules[1], len(rules[1])))
                generated_rules.append(function[0])
                pbar.update(1)
    
    # Save results
    logger.info("Saving generated rules and metadata...")
    output_prefix = f'{job_index}_seed_{seed}_{args.class_type}_{args.quantity}'
    rules_filename = os.path.join(rules_path, f'{output_prefix}_rules.pkl')
    metadata_filename = os.path.join(metadata_path, f'{output_prefix}_metadata.pkl')
    
    try:
        RuleStorage.write_rules(generated_rules, rules_filename)
        RuleStorage.write_metadata(rules_metadata, metadata_filename)
        logger.info(f"Successfully saved rules to {rules_filename}")
        logger.info(f"Successfully saved metadata to {metadata_filename}")
    except Exception as e:
        logger.error(f"Error saving output files: {str(e)}")
        sys.exit(1)
    
    logger.info('Rule generation completed successfully')

if __name__ == "__main__":
    main() 


#### Example usage #########
# Run positive rules generation
# python auto_rules.py \
# --nodes-path "${NODES_FILE}" \
# --source-target-path "${SOURCE_TARGET_FILE}" \
# --labels-dict-path "${LABELS_DICT_FILE}" \
# --output-dir "${OUTPUT_DIR}" \
# --log-dir "${LOG_DIR}" \
#  --class-type positive \
#  --quantity "${QUANTITY}" \
#  --seed "${SEED}"

# Run negative rules generation
#python auto_rules.py \
#  --nodes-path "${NODES_FILE}" \
#  --source-target-path "${SOURCE_TARGET_FILE}" \
#  --labels-dict-path "${LABELS_DICT_FILE}" \
#  --output-dir "${OUTPUT_DIR}" \
#  --log-dir "${LOG_DIR}" \
#  --class-type negative \
#  --quantity "${QUANTITY}" \
#  --seed "${SEED}"
