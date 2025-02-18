import pandas as pd
import joblib
import logging
from datetime import datetime
import os

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
    """Parse command line arguments for clinical data processing"""
    parser = argparse.ArgumentParser(description='Process clinical data with time windows and create tuple lists')
    
    # Required arguments
    parser.add_argument('--time-df', required=True,
                      help='Path to time DataFrame file')
    parser.add_argument('--labs-path', required=True,
                      help='Path to labs data file')
    parser.add_argument('--vitals-path', required=True,
                      help='Path to vitals data file')
    parser.add_argument('--meds-path', required=True,
                      help='Path to medications data file')
    parser.add_argument('--output-dir', required=True,
                      help='Directory for saving output files')
    parser.add_argument('--log-dir', required=True,
                      help='Directory for saving log files')
    
    # Optional arguments
    parser.add_argument('--run-name', default='clinical_processing',
                      help='Name for this run (used in logging)')
    parser.add_argument('--semantic', action='store_true',
                      help='Whether to create semantic tuple lists')
    parser.add_argument('--fixed-date', default='2003-09-12 12:00:00',
                      help='Fixed reference date for time alignment')
    
    return parser.parse_args()

def filter_clinical_data(time_df_path: str,
                        labs_path: str,
                        vitals_path: str,
                        meds_path: str,
                        output_dir: str,
                        logger: logging.Logger):
    """
    Filter clinical data based on MRNs in the time file
    
    Args:
        time_df_path: Path to the time DataFrame file
        labs_path: Path to the labs data
        vitals_path: Path to the vitals data
        meds_path: Path to the medications data
        output_dir: Directory to save filtered outputs
        logger: Logger instance
    """
    # Load time DataFrame
    logger.info("Loading time DataFrame...")
    time_df = pd.read_csv(time_df_path)
    valid_mrns = set(time_df['MRN'])
    logger.info(f"Found {len(valid_mrns)} unique MRNs in time file")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter Labs
    logger.info("Processing Labs data...")
    labs_df = joblib.load(labs_path)
    filtered_labs = labs_df[labs_df['MRN'].isin(valid_mrns)]
    logger.info(f"Filtered labs from {len(labs_df)} to {len(filtered_labs)} records")
    joblib.dump(filtered_labs, os.path.join(output_dir, 'filtered_labs.bz2'), compress=('bz2', 3))
    
    # Filter Vitals
    logger.info("Processing Vitals data...")
    vitals_df = joblib.load(vitals_path)
    filtered_vitals = vitals_df[vitals_df['MRN'].isin(valid_mrns)]
    logger.info(f"Filtered vitals from {len(vitals_df)} to {len(filtered_vitals)} records")
    joblib.dump(filtered_vitals, os.path.join(output_dir, 'filtered_vitals.bz2'), compress=('bz2', 3))
    
    # Filter Medications
    logger.info("Processing Medications data...")
    meds_df = joblib.load(meds_path)
    filtered_meds = meds_df[meds_df['MRN'].isin(valid_mrns)]
    logger.info(f"Filtered medications from {len(meds_df)} to {len(filtered_meds)} records")
    joblib.dump(filtered_meds, os.path.join(output_dir, 'filtered_meds.bz2'), compress=('bz2', 3))
    
    # Generate summary statistics
    summary = {
        'total_mrns': len(valid_mrns),
        'labs_records': len(filtered_labs),
        'labs_unique_mrns': filtered_labs['MRN'].nunique(),
        'vitals_records': len(filtered_vitals),
        'vitals_unique_mrns': filtered_vitals['MRN'].nunique(),
        'meds_records': len(filtered_meds),
        'meds_unique_mrns': filtered_meds['MRN'].nunique()
    }
    
    # Save summary
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'filtering_summary.csv'), index=False)
    logger.info("Filtering complete. Summary saved to output directory.")
    
    return summary

def add_time_flags(df: pd.DataFrame, 
                    datetime_col: str,
                    patient_start_col: str = 'patient_start_time',
                    anesthesia_start_col: str = 'anesthesia_start_time',
                    fixed_date: datetime = datetime(2003, 9, 12, 12, 0)) -> pd.DataFrame:
    """
    Add time-based flags and calculations to the clinical data DataFrame.
    
    Args:
        df: Input DataFrame with clinical data
        datetime_col: Name of the column containing event timestamps
        patient_start_col: Name of the column containing patient start times
        anesthesia_start_col: Name of the column containing anesthesia start times
        fixed_date: Reference date for time alignment
        
    Returns:
        DataFrame with added time flags and calculations
    """
    # Ensure datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[patient_start_col] = pd.to_datetime(df[patient_start_col])
    df[anesthesia_start_col] = pd.to_datetime(df[anesthesia_start_col])
    
    # Calculate time elapsed
    df['time_elapsed'] = df[datetime_col] - df[patient_start_col]
    
    # Add pre-op flag based on anesthesia start time
    df['pre_op_flag'] = df.apply(
        lambda x: 1 if pd.notna(x[anesthesia_start_col]) and 
                    x[datetime_col] <= x[anesthesia_start_col] else 0,
        axis=1
    )
    
    # Add post-start flag based on patient start time
    df['post_start_flag'] = df.apply(
        lambda x: 1 if pd.notna(x[patient_start_col]) and 
                    x[datetime_col] >= x[patient_start_col] else 0,
        axis=1
    )
    
    # Add aligned datetime
    fixed_date = pd.Timestamp(fixed_date)
    df['aligned_datetime'] = fixed_date + df['time_elapsed']
    
    # Create within_window flag
    df['within_window'] = ((df['post_start_flag'] == 1) & 
                          (df['pre_op_flag'] == 1)).astype(int)
    
    return df

def process_clinical_data_with_times(labs_df: pd.DataFrame = None,
                                   vitals_df: pd.DataFrame = None,
                                   meds_df: pd.DataFrame = None,
                                   output_dir: str = None,
                                   logger: logging.Logger = None) -> dict:
    """
    Process clinical data with time flags and save results.
    
    Args:
        labs_df: Labs DataFrame
        vitals_df: Vitals DataFrame
        meds_df: Medications DataFrame
        output_dir: Directory to save processed outputs
        logger: Logger instance
    
    Returns:
        Dictionary with processing summary
    """
    results = {}
    
    if labs_df is not None:
        logger.info("Processing labs with time flags...")
        labs_processed = add_time_flags(
            labs_df.copy(),
            datetime_col='Collection Datetime'
        )
        results['labs'] = {
            'total_records': len(labs_processed),
            'within_window': len(labs_processed[labs_processed['within_window'] == 1])
        }
        if output_dir:
            joblib.dump(labs_processed, 
                       os.path.join(output_dir, 'labs_with_timeflags.bz2'),
                       compress=('bz2', 3))
    
    if vitals_df is not None:
        logger.info("Processing vitals with time flags...")
        vitals_processed = add_time_flags(
            vitals_df.copy(),
            datetime_col='RECORDED_TIME'
        )
        results['vitals'] = {
            'total_records': len(vitals_processed),
            'within_window': len(vitals_processed[vitals_processed['within_window'] == 1])
        }
        if output_dir:
            joblib.dump(vitals_processed,
                       os.path.join(output_dir, 'vitals_with_timeflags.bz2'),
                       compress=('bz2', 3))
    
    if meds_df is not None:
        logger.info("Processing medications with time flags...")
        meds_processed = add_time_flags(
            meds_df.copy(),
            datetime_col='MED_ACTION_TIME'
        )
        results['meds'] = {
            'total_records': len(meds_processed),
            'within_window': len(meds_processed[meds_processed['within_window'] == 1])
        }
        if output_dir:
            joblib.dump(meds_processed,
                       os.path.join(output_dir, 'meds_with_timeflags.bz2'),
                       compress=('bz2', 3))
    
    if output_dir:
        pd.DataFrame(results).to_csv(os.path.join(output_dir, 'time_processing_summary.csv'))
    
    return results

def create_tuple_lists_basic(df: pd.DataFrame, 
                           name_col: str = 'target_name',
                           time_col: str = 'aligned_datetime') -> pd.DataFrame:
    """
    Create basic tuple lists (name, timestamp) for each MRN.
    """
    def create_basic_tuple(row):
        return (row[name_col], row[time_col])
    
    df['BasicTuple'] = df.apply(create_basic_tuple, axis=1)
    
    result = df.groupby('MRN').apply(
        lambda group: pd.Series({
            'TupleList': list(dict.fromkeys(group['BasicTuple'])),
            'Count': len(dict.fromkeys(group['BasicTuple']))
        })
    ).reset_index()
    
    return result

def create_tuple_lists_semantic(df: pd.DataFrame, 
                              data_type: str,
                              name_col: str = 'target_name',
                              time_col: str = 'aligned_datetime',
                              interval_col: str = 'Interval Index') -> pd.DataFrame:
    """
    Create semantic tuple lists (name_kde_interval, timestamp, original_name) for each MRN.
    """
    def create_semantic_tuple(row):
        if data_type in ['vitals', 'labs']:
            semantic_name = f"{row[name_col]}_kde_{row[interval_col]}"
            return (semantic_name, row[time_col], row[name_col])
        else:  # medications
            return (row[name_col], row[time_col], 'medication')
    
    df['SemanticTuple'] = df.apply(create_semantic_tuple, axis=1)
    
    result = df.groupby('MRN').apply(
        lambda group: pd.Series({
            'SemanticTupleList': list(dict.fromkeys(group['SemanticTuple'])),
            'CountSem': len(dict.fromkeys(group['SemanticTuple']))
        })
    ).reset_index()
    
    return result

def combine_tuple_lists(dataframes: List[pd.DataFrame], semantic: bool = False) -> pd.DataFrame:
    """
    Combine tuple lists from multiple dataframes.
    
    Args:
        dataframes: List of DataFrames with tuple lists
        semantic: Whether to use semantic tuples or basic tuples
    """
    combined_tuples: Dict[str, List[Tuple]] = {}
    
    list_col = 'SemanticTupleList' if semantic else 'TupleList'
    
    for df in dataframes:
        for _, row in df.iterrows():
            mrn = row['MRN']
            tuples = row[list_col]
            
            if mrn in combined_tuples:
                combined_tuples[mrn].extend(tuples)
            else:
                combined_tuples[mrn] = tuples.copy()
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(list(combined_tuples.items()), 
                             columns=['MRN', f'CombinedTupleList{"Sem" if semantic else ""}'])
    
    # Sort by timestamp
    combined_df[f'CombinedTupleList{"Sem" if semantic else ""}'] = \
        combined_df[f'CombinedTupleList{"Sem" if semantic else ""}'].apply(
            lambda x: sorted(x, key=lambda tup: tup[1])
        )
    
    # Add count
    combined_df[f'CombinedCount{"Sem" if semantic else ""}'] = \
        combined_df[f'CombinedTupleList{"Sem" if semantic else ""}'].apply(len)
    
    return combined_df

def process_clinical_tuples(labs_df: pd.DataFrame = None,
                          vitals_df: pd.DataFrame = None,
                          meds_df: pd.DataFrame = None,
                          output_dir: str = None,
                          semantic: bool = False) -> pd.DataFrame:
    """
    Process clinical data into tuple lists.
    
    Args:
        labs_df: Laboratory results DataFrame
        vitals_df: Vital signs DataFrame
        meds_df: Medications DataFrame
        output_dir: Directory to save outputs
        semantic: Whether to create semantic or basic tuples
    """
    processed_dfs = []
    
    # Process each dataframe if provided
    for df, name, type_ in [
        (labs_df, 'labs', 'labs'),
        (vitals_df, 'vitals', 'vitals'),
        (meds_df, 'meds', 'meds')
    ]:
        if df is not None:
            if semantic:
                result = create_tuple_lists_semantic(df, data_type=type_)
            else:
                result = create_tuple_lists_basic(df)
            
            processed_dfs.append(result)
            
            if output_dir:
                result.to_csv(
                    f"{output_dir}/{name}_{'semantic_' if semantic else ''}tuples.csv",
                    index=False
                )
    
    # Combine all processed dataframes
    combined_result = combine_tuple_lists(processed_dfs, semantic=semantic)
    
    if output_dir:
        combined_result.to_csv(
            f"{output_dir}/combined_{'semantic_' if semantic else ''}tuples.csv",
            index=False
        )
    
    return combined_result

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logger(args.log_dir)
    logger.info(f"Starting clinical data processing run: {args.run_name}")
    
    try:
        # Step 1: Filter data
        summary = filter_clinical_data(
            time_df_path=args.time_df,
            labs_path=args.labs_path,
            vitals_path=args.vitals_path,
            meds_path=args.meds_path,
            output_dir=args.output_dir,
            logger=logger
        )
        logger.info("Filtering complete")
        logger.info(f"Summary: {summary}")
        
        # Step 2: Load filtered data
        labs_df = joblib.load(os.path.join(args.output_dir, 'filtered_labs.bz2'))
        vitals_df = joblib.load(os.path.join(args.output_dir, 'filtered_vitals.bz2'))
        meds_df = joblib.load(os.path.join(args.output_dir, 'filtered_meds.bz2'))
        
        # Step 3: Process with time flags
        logger.info("Processing time flags")
        time_results = process_clinical_data_with_times(
            labs_df=labs_df,
            vitals_df=vitals_df,
            meds_df=meds_df,
            output_dir=args.output_dir,
            logger=logger
        )
        logger.info(f"Time processing results: {time_results}")
        
        # Step 4: Load time-processed data
        labs_processed = joblib.load(os.path.join(args.output_dir, 'labs_with_timeflags.bz2'))
        vitals_processed = joblib.load(os.path.join(args.output_dir, 'vitals_with_timeflags.bz2'))
        meds_processed = joblib.load(os.path.join(args.output_dir, 'meds_with_timeflags.bz2'))
        
        # Step 5: Create tuple lists
        logger.info(f"Creating {'semantic' if args.semantic else 'basic'} tuple lists")
        tuple_results = process_clinical_tuples(
            labs_df=labs_processed[labs_processed['within_window'] == 1],
            vitals_df=vitals_processed[vitals_processed['within_window'] == 1],
            meds_df=meds_processed[meds_processed['within_window'] == 1],
            output_dir=args.output_dir,
            semantic=args.semantic
        )
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
