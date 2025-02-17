import os
import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def setup_logger(log_dir):
    """Set up and return a configured logger"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create unique log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f'data_processing_{timestamp}.log')

    # Create handlers
    file_handler = logging.FileHandler(log_file_path, mode='a')
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def filter_by_mrn(label_df_path, data_dir, output_dir, logger):
    """Filter CSV files by MRN"""
    logger.info("Starting MRN filtering step")
    
    try:
        # Load and process MRN list
        label_df = pd.read_csv(label_df_path)
        included_mrns = set(label_df['INCLUDED_MRNs'].dropna().unique())
        logger.info(f"Loaded {len(included_mrns)} unique MRNs for filtering")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each CSV file in the target directory
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.csv'):
                csv_path = os.path.join(data_dir, file_name)
                try:
                    logger.info(f"Processing {file_name}")
                    df = pd.read_csv(csv_path)
                    
                    if 'MRN' in df.columns:
                        # Filter DataFrame
                        initial_rows = len(df)
                        filtered_df = df[df['MRN'].isin(included_mrns)]
                        final_rows = len(filtered_df)
                        
                        # Save filtered DataFrame
                        output_file = os.path.join(output_dir, file_name)
                        filtered_df.to_csv(output_file, index=False)
                        
                        logger.info(f"Filtered {file_name}: {initial_rows} -> {final_rows} rows")
                    else:
                        logger.warning(f"No MRN column found in {file_name}")
                
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {str(e)}")
                    continue
        
        logger.info("MRN filtering completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in MRN filtering process: {str(e)}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Filter medical records by MRN.')
    
    # Required arguments
    parser.add_argument('--label_df_path', 
                        required=True,
                        help='Path to the CSV file containing MRNs to include')
    
    parser.add_argument('--data_dir', 
                        required=True,
                        help='Directory containing the source CSV files')
    
    parser.add_argument('--output_dir', 
                        required=True,
                        help='Directory where filtered files will be saved')
    
    parser.add_argument('--log_dir', 
                        required=True,
                        help='Directory where log files will be saved')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    logger.info("Starting data processing pipeline with arguments:")
    logger.info(f"Label DF Path: {args.label_df_path}")
    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Log Directory: {args.log_dir}")
    
    # Run the MRN filtering
    success = filter_by_mrn(
        args.label_df_path,
        args.data_dir,
        args.output_dir,
        logger
    )
    
    if success:
        logger.info("MRN filtering step completed. Ready for BP processing step.")
    else:
        logger.error("MRN filtering failed. Please check the logs and fix any issues before proceeding.") 


########## Example usage ##################
# python filter_mrns.py \
#    --label_df_path ./data/mrn_list.csv \
#    --data_dir ./data/raw/ \
#    --output_dir ./data/filtered/ \
#   --log_dir ./logs/
