import os
import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from typing import Dict, Union, List, Tuple

def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    """Set up and return a configured logger"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{run_name}.log')
    
    # Create handlers
    file_handler = logging.FileHandler(log_file_path, mode='a')
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_conversion_dict(conversion_file: str, source_col: str, target_col: str, unmatch_val: str = 'unmatched') -> Dict:
    """Create and return a conversion dictionary from a CSV file"""
    conversion_df = pd.read_csv(conversion_file)
    conversion_tuples = [(source, target) for source, target in 
                        zip(conversion_df[source_col], conversion_df[target_col]) 
                        if source != unmatch_val]
    conversion_dict = dict(conversion_tuples)
    filtered_dict = {k: v for k, v in conversion_dict.items() if v != unmatch_val}
    return {v: k for k, v in filtered_dict.items()}

def process_vitals(vitals_file: str, conversion_file: str, output_dir: str, logger: logging.Logger) -> str:
    """Process vitals data including BP splitting and standardization"""
    logger.info("Starting vitals processing")
    
    # Load vitals data
    vitals_df = pd.read_csv(vitals_file)
    logger.info(f"Loaded vitals data with {len(vitals_df)} records")
    
    # Process BP readings
    bp_rows = vitals_df[vitals_df['FLO_DISPLAY_NAME'] == 'BP'].copy()
    bp_rows[['Systolic', 'Diastolic']] = bp_rows['MEAS_VALUE'].str.split('/', expand=True)
    
    # Create separate systolic and diastolic records
    systolic_df = bp_rows[['FLO_DISPLAY_NAME', 'Systolic']].copy()
    systolic_df['FLO_DISPLAY_NAME'] = 'BLOOD PRESSURE - systolic'
    systolic_df = systolic_df.rename(columns={'Systolic': 'MEAS_VALUE'})
    
    diastolic_df = bp_rows[['FLO_DISPLAY_NAME', 'Diastolic']].copy()
    diastolic_df['FLO_DISPLAY_NAME'] = 'BLOOD PRESSURE - diastolic'
    diastolic_df = diastolic_df.rename(columns={'Diastolic': 'MEAS_VALUE'})
    
    # Combine processed data
    vitals_df = vitals_df[vitals_df['FLO_DISPLAY_NAME'] != 'BP']
    processed_df = pd.concat([vitals_df, systolic_df, diastolic_df], ignore_index=True)
    
    # Add conversion mapping
    conversion_dict = create_conversion_dict(conversion_file, 'sinai_vital', 'mover_vital')
    processed_df['convert_sinai'] = processed_df['FLO_DISPLAY_NAME'].apply(
        lambda x: conversion_dict.get(x, 'not_matched'))
    
    # Save processed data
    output_file = os.path.join(output_dir, 'processed_vitals.bz2')
    joblib.dump(processed_df, output_file, compress=('bz2', 3))
    
    return output_file

def clean_vitals_data(df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Clean vitals data using summary statistics"""
    df['MEAS_VALUE'] = pd.to_numeric(df['MEAS_VALUE'], errors='coerce')
    
    for _, row in summary_df.iterrows():
        vital_name = row['Vitals Name']
        if vital_name in df['convert_sinai'].unique():
            mask = df['convert_sinai'] == vital_name
            df.loc[mask, 'clean_val'] = df.loc[mask, 'MEAS_VALUE'].clip(
                lower=row['1%'], upper=row['99%'])
    
    return df

def generate_summary_stats(data: pd.Series) -> Dict:
    """Generate summary statistics for a data series"""
    return {
        'Count': data.count(),
        'Mean': data.mean(),
        'Std': data.std(),
        'Min': data.min(),
        '1%': data.quantile(0.01),
        '5%': data.quantile(0.05),
        '25%': data.quantile(0.25),
        '50%': data.median(),
        '75%': data.quantile(0.75),
        '95%': data.quantile(0.95),
        '99%': data.quantile(0.99),
        'Max': data.max()
    }

def create_histograms(df: pd.DataFrame, output_dir: str, value_col: str, name_col: str, logger: logging.Logger):
    """Create and save histograms for each unique measurement"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name in df[name_col].unique():
        data = pd.to_numeric(df[df[name_col] == name][value_col], errors='coerce').dropna()
        
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, color='blue', alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of {name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        output_path = os.path.join(output_dir, f'{name}_histogram.png')
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f'Created histogram for {name}')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process clinical data including vitals and labs.')
    
    # Existing arguments
    parser.add_argument('--vitals_file', required=True,
                        help='Path to vitals data CSV')
    parser.add_argument('--vitals_conversion', required=True,
                        help='Path to vitals conversion CSV')
    parser.add_argument('--labs_file', required=True,
                        help='Path to labs data CSV')
    parser.add_argument('--labs_conversion', required=True,
                        help='Path to labs conversion CSV')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for output files')
    parser.add_argument('--log_dir', required=True,
                        help='Directory for log files')
    
    # Add new arguments for interval files
    parser.add_argument('--labs_intervals', required=True,
                       help='Path to labs intervals PKL file')
    parser.add_argument('--vitals_intervals', required=True,
                       help='Path to vitals intervals PKL file')
    parser.add_argument('--vitals_summary', required=True,
                       help='Path to vitals summary statistics CSV')
    
    return parser.parse_args()

class ClinicalLabStandardizer:
    """Handles standardization of clinical lab measurements"""
    def __init__(self):
        # Define standard units for each lab test
        self.standard_units = {
            'Albumin': 'g/dL',
            'Creatinine': 'mg/dL',
            'Troponin I.cardiac': 'ng/mL',
            'Leukocytes': 'K/µL',
            'Magnesium': 'mg/dL',
            'Protein': 'g/dL'
        }
        
        # Define unit conversions
        self.conversion_factors = {
            'Albumin': {
                'MG/L': lambda x: x / 10000,  # mg/L to g/dL
                'G/TOT VOL': lambda x: x      # already in g/dL
            },
            'Creatinine': {
                'G/TOT VOL': lambda x: x * 100  # g/dL to mg/dL
            },
            'Troponin I.cardiac': {
                'ng/L': lambda x: x / 1000    # ng/L to ng/mL
            }
        }
        
        # Define different measurement types
        self.different_measurement_types = {
            'Albumin': {
                'different_units': ['%', 'MCG/MIN'],
                'reason': 'Different measurement methodology'
            },
            'Creatinine': {
                'different_units': ['mg/d', 'MG/24HR'],
                'reason': '24-hour collection measurement'
            },
            'Leukocytes': {
                'different_units': ['#/HPF'],
                'reason': 'Semi-quantitative microscopic measurement'
            },
            'Magnesium': {
                'different_units': ['MG/24 HRS'],
                'reason': '24-hour collection measurement'
            },
            'Protein': {
                'different_units': ['MG/DAY'],
                'reason': '24-hour collection measurement'
            }
        }
        
        # Define equivalent units
        self.equivalent_units = {
            'MMOL/L': ['MMOL/TOT VOL', 'MEQ/L', 'mmol/L'],
            'MG/DL': ['MG/TV', 'mg/dL', 'MG/DL'],
            'K/µL': ['THOUS/MCL', 'THOUS/ CU MM'],
            'G/DL': ['g/dL', 'G/DL']
        }

    def is_different_measurement_type(self, lab_name: str, unit: str) -> Dict[str, Union[bool, str]]:
        """Check if a unit represents a different measurement type."""
        if lab_name in self.different_measurement_types:
            if unit.upper() in [u.upper() for u in self.different_measurement_types[lab_name]['different_units']]:
                return {
                    'is_different': True,
                    'reason': self.different_measurement_types[lab_name]['reason']
                }
        return {
            'is_different': False,
            'reason': ''
        }

    def standardize_unit(self, lab_name: str, original_unit: str, value: float) -> Dict[str, Union[str, float, bool]]:
        """Standardize a lab value to the preferred unit."""
        measurement_check = self.is_different_measurement_type(lab_name, original_unit)
        if measurement_check['is_different']:
            return {
                'standardized_value': value,
                'standardized_unit': original_unit,
                'converted': False,
                'different_measurement_type': True,
                'notes': measurement_check['reason']
            }

        if lab_name not in self.standard_units:
            return {
                'standardized_value': value,
                'standardized_unit': original_unit,
                'converted': False,
                'different_measurement_type': False,
                'notes': 'Lab test not found in standardization map'
            }
        
        standard_unit = self.standard_units[lab_name]
        
        # Check for equivalent units
        for std, equivalents in self.equivalent_units.items():
            if original_unit.upper() in [u.upper() for u in equivalents + [std]]:
                return {
                    'standardized_value': value,
                    'standardized_unit': standard_unit,
                    'converted': False,
                    'different_measurement_type': False,
                    'notes': 'Equivalent unit notation'
                }

        # Apply conversion if needed
        if lab_name in self.conversion_factors and original_unit in self.conversion_factors[lab_name]:
            converted_value = self.conversion_factors[lab_name][original_unit](value)
            return {
                'standardized_value': converted_value,
                'standardized_unit': standard_unit,
                'converted': True,
                'different_measurement_type': False,
                'notes': f'Converted from {original_unit} to {standard_unit}'
            }

        return {
            'standardized_value': value,
            'standardized_unit': original_unit,
            'converted': False,
            'different_measurement_type': False,
            'notes': 'No conversion factor available'
        } 



    def process_labs_dataframe(self, df: pd.DataFrame, drop_different_measurements: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Process and standardize laboratory results DataFrame while tracking affected records.
        
        Args:
            df: DataFrame with lab results
            drop_different_measurements: If True, removes rows with different measurement types
        
        Returns:
            Tuple of (processed DataFrame, affected records dictionary)
        """
        result_df = df.copy()
        
        # Initialize new columns
        result_df['standardized_value'] = None
        result_df['standardized_unit'] = None
        result_df['conversion_applied'] = False
        result_df['different_measurement_type'] = False
        result_df['conversion_notes'] = ''
        result_df['processing_datetime'] = datetime.now()
        
        # Track affected records
        affected_records = {
            'converted': {'MRNs': set(), 'LOG_IDs': set()},
            'dropped': {'MRNs': set(), 'LOG_IDs': set()},
            'errors': {'MRNs': set(), 'LOG_IDs': set()}
        }
        
        for idx, row in df.iterrows():
            try:
                # Handle non-numeric values
                try:
                    value = float(row['Observation Value'])
                except (ValueError, TypeError):
                    result_df.at[idx, 'conversion_notes'] = f'Non-numeric value: {row["Observation Value"]}'
                    affected_records['errors']['MRNs'].add(row['MRN'])
                    affected_records['errors']['LOG_IDs'].add(row['LOG_ID'])
                    continue
                
                # Standardize the unit
                result = self.standardize_unit(
                    str(row['Lab Name']),
                    str(row['Measurement Units']),
                    value
                )
                
                # Update result DataFrame
                result_df.at[idx, 'standardized_value'] = result['standardized_value']
                result_df.at[idx, 'standardized_unit'] = result['standardized_unit']
                result_df.at[idx, 'conversion_applied'] = result['converted']
                result_df.at[idx, 'different_measurement_type'] = result['different_measurement_type']
                result_df.at[idx, 'conversion_notes'] = result['notes']
                
                # Track affected records
                if result['converted']:
                    affected_records['converted']['MRNs'].add(row['MRN'])
                    affected_records['converted']['LOG_IDs'].add(row['LOG_ID'])
                if result['different_measurement_type']:
                    affected_records['dropped']['MRNs'].add(row['MRN'])
                    affected_records['dropped']['LOG_IDs'].add(row['LOG_ID'])
                
            except Exception as e:
                result_df.at[idx, 'conversion_notes'] = f'Error processing row: {str(e)}'
                affected_records['errors']['MRNs'].add(row['MRN'])
                affected_records['errors']['LOG_IDs'].add(row['LOG_ID'])
        
        # Generate summary of affected records
        affected_summary = {
            'category': [],
            'unique_mrns': [],
            'unique_log_ids': [],
            'total_records': []
        }
        
        for category, records in affected_records.items():
            affected_summary['category'].append(category)
            affected_summary['unique_mrns'].append(len(records['MRNs']))
            affected_summary['unique_log_ids'].append(len(records['LOG_IDs']))
            affected_summary['total_records'].append(
                len(result_df[
                    (result_df['MRN'].isin(records['MRNs'])) & 
                    (result_df['LOG_ID'].isin(records['LOG_IDs']))
                ])
            )
        
        affected_summary_df = pd.DataFrame(affected_summary)
        
        # Remove different measurement types if requested
        if drop_different_measurements:
            result_df = result_df[~result_df['different_measurement_type']]
        
        return result_df, affected_summary_df 

    def generate_lab_processing_report(self, df: pd.DataFrame, affected_summary_df: pd.DataFrame, output_dir: str):
        """
        Generate a detailed report of lab processing results.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(output_dir, f'lab_processing_report_{timestamp}')
        os.makedirs(report_dir, exist_ok=True)
        
        # Save affected records summary
        affected_summary_df.to_csv(os.path.join(report_dir, 'affected_records_summary.csv'), index=False)
        
        # Generate lab-specific summaries
        lab_summaries = []
        for lab_name in df['Lab Name'].unique():
            lab_df = df[df['Lab Name'] == lab_name]
            
            summary = {
                'Lab Name': lab_name,
                'Total Records': len(lab_df),
                'Converted Records': sum(lab_df['conversion_applied']),
                'Different Measurement Types': sum(lab_df['different_measurement_type']),
                'Original Units': ', '.join(sorted(lab_df['Measurement Units'].unique())),
                'Conversion Success Rate': f"{(sum(lab_df['conversion_applied'])/len(lab_df)*100):.1f}%"
            }
            lab_summaries.append(summary)
        
        lab_summary_df = pd.DataFrame(lab_summaries)
        lab_summary_df.to_csv(os.path.join(report_dir, 'lab_processing_summary.csv'), index=False)
        
        return report_dir


def assign_interval_index_vectorized(lab_value: float, intervals: List[Tuple]) -> int:
    """
    Assign interval index for lab values using KDE-based intervals.
    
    Args:
        lab_value: Lab measurement value
        intervals: List of interval tuples from KDE analysis
        
    Returns:
        int: Index of interval (1-based) or length of intervals if above last interval
    """
    # Below first interval
    if lab_value < intervals[0][0][0]:
        return 1
        
    # Check each interval
    for i, interval in enumerate(intervals):
        lower_bound = interval[0][0]
        upper_bound = interval[1][0]
        if lower_bound <= lab_value <= upper_bound:
            return i + 1
            
    # Above last interval
    return len(intervals)

def assign_interval_index_vectorized_vitals(vital_value: float, intervals: List[Tuple]) -> int:
    """
    Assign interval index for vital signs using KDE-based intervals.
    
    Args:
        vital_value: Vital sign measurement value
        intervals: List of interval tuples from KDE analysis
        
    Returns:
        int: Index of interval (1-based) or length of intervals if above last interval
    """
    # Below first interval
    if vital_value < intervals[0][0][0]:
        return 1
        
    # Check each interval    
    for i, interval in enumerate(intervals):
        lower_bound = interval[0][0]
        upper_bound = interval[1][0]
        if lower_bound <= vital_value <= upper_bound:
            return i + 1
            
    # Above last interval    
    return len(intervals)

def process_labs_intervals(labs_df: pd.DataFrame, labs_int: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Process labs data with KDE-based intervals.
    
    Args:
        labs_df: Lab results DataFrame
        labs_int: DataFrame with KDE intervals 
        logger: Logger instance
        
    Returns:
        DataFrame with interval assignments
    """
    # Create intervals dictionary with normalized keys
    intervals_dict = {
        row['Name'].strip().lower(): row['Intervals'] 
        for _, row in labs_int.iterrows()
    }
    
    # Assign intervals using vectorized function
    labs_df['Interval Index'] = labs_df.apply(
        lambda row: assign_interval_index_vectorized(
            row['standardized_value'], 
            intervals_dict.get(row['convert_sinai'].lower(), [])
        ) if row['convert_sinai'].lower() in intervals_dict else np.nan,
        axis=1
    )
    
    return labs_df

def process_vitals_intervals(vitals_df: pd.DataFrame, vitals_int: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Process vitals data with KDE-based intervals.
    
    Args:
        vitals_df: Vitals DataFrame
        vitals_int: DataFrame with KDE intervals
        logger: Logger instance
        
    Returns:
        DataFrame with interval assignments
    """
    # Create intervals dictionary with normalized keys
    intervals_dict = {
        row['Name'].strip().lower(): row['Intervals'] 
        for _, row in vitals_int.iterrows()
    }
    
    # Assign intervals using vectorized function
    vitals_df['Interval Index'] = vitals_df.apply(
        lambda row: assign_interval_index_vectorized_vitals(
            row['clean_val'],
            intervals_dict.get(row['convert_sinai'].lower(), [])
        ) if row['convert_sinai'].lower() in intervals_dict else np.nan,
        axis=1
    )
    
    return vitals_df

def clean_vitals(df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean vitals data using summary statistics.
    
    Args:
        df: Input DataFrame with vitals measurements
        summary_df: DataFrame with summary statistics for each vital
        
    Returns:
        DataFrame with cleaned values
    """
    df['MEAS_VALUE'] = pd.to_numeric(df['MEAS_VALUE'], errors='coerce')
    
    for _, row in summary_df.iterrows():
        vital_name = row['Vitals Name']
        if vital_name in df['convert_sinai'].unique():
            mask = df['convert_sinai'] == vital_name
            df.loc[mask, 'clean_val'] = df.loc[mask, 'MEAS_VALUE'].clip(
                lower=row['1%'], 
                upper=row['99%']
            )
    
    return df

def process_labs(labs_file: str, labs_conversion: str, output_dir: str, logger: logging.Logger) -> Dict[str, str]:
    """Process lab data and generate reports"""
    logger.info("Starting lab processing")
    
    # Load labs data
    labs_df = pd.read_csv(labs_file)
    logger.info(f"Loaded labs data with {len(labs_df)} records")
    
    # Initialize standardizer
    standardizer = ClinicalLabStandardizer()
    
    # Process labs
    processed_df, affected_summary = standardizer.process_labs_dataframe(labs_df)
    
    # Generate report
    report_dir = standardizer.generate_lab_processing_report(
        processed_df, 
        affected_summary,
        output_dir
    )
    
    # Save processed data
    output_file = os.path.join(output_dir, 'processed_labs.bz2')
    joblib.dump(processed_df, output_file, compress=('bz2', 3))
    
    return {
        'processed_file': output_file,
        'report_dir': report_dir
    }

def process_vital_signs(vitals_path: str, output_dir: str, logger) -> str:
    """Process and clean vital signs data"""
    logger.info("Starting vital signs processing")
    
    # Load vitals data
    vitals_df = pd.read_csv(vitals_path)
    
    # Calculate thresholds for each vital sign
    vital_stats = {}
    for vital in vitals_df['FLO_DISPLAY_NAME'].unique():
        data = pd.to_numeric(vitals_df[vitals_df['FLO_DISPLAY_NAME'] == vital]['MEAS_VALUE'], 
                           errors='coerce')
        vital_stats[vital] = {
            'p1': data.quantile(0.01),
            'p99': data.quantile(0.99)
        }
    
    # Clean data using thresholds
    cleaned_data = []
    for _, row in vitals_df.iterrows():
        try:
            value = float(row['MEAS_VALUE'])
            vital = row['FLO_DISPLAY_NAME']
            if vital in vital_stats:
                if value >= vital_stats[vital]['p1'] and value <= vital_stats[vital]['p99']:
                    cleaned_data.append(row)
        except (ValueError, TypeError):
            continue
    
    result_df = pd.DataFrame(cleaned_data)
    
    # Create and save histograms
    plot_dir = os.path.join(output_dir, 'histograms')
    os.makedirs(plot_dir, exist_ok=True)
    
    for vital in result_df['FLO_DISPLAY_NAME'].unique():
        data = pd.to_numeric(result_df[result_df['FLO_DISPLAY_NAME'] == vital]['MEAS_VALUE'], 
                           errors='coerce')
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50)
        plt.title(f'Distribution of {vital}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(plot_dir, f'{vital}_histogram.png'))
        plt.close()
    
    # Save processed data
    output_path = os.path.join(output_dir, 'cleaned_vitals.csv')
    result_df.to_csv(output_path, index=False)
    
    logger.info(f'Vital signs processing complete. Output saved to: {output_path}')
    return output_path

if __name__ == "__main__":
    args = parse_arguments()
    logger = setup_logger(args.log_dir, "clinical_data_processing")
    
    try:
        # Verify all input files exist
        required_files = [
            args.vitals_file,
            args.vitals_conversion,
            args.labs_file,
            args.labs_conversion,
            args.labs_intervals,
            args.vitals_intervals,
            args.vitals_summary
        ]
        for input_file in required_files:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process vitals (includes BP processing)
        logger.info("Step 1: Processing vitals data")
        vitals_output = process_vitals(
            args.vitals_file, 
            args.vitals_conversion,
            args.output_dir,
            logger
        )
        
        # Process labs
        logger.info("Step 2: Processing lab results")
        labs_results = process_labs(
            args.labs_file,
            args.labs_conversion,
            args.output_dir,
            logger
        )
        
        # Load interval data
        logger.info("Step 3: Loading interval data")
        labs_int = joblib.load(args.labs_intervals)
        vitals_int = joblib.load(args.vitals_intervals)
        summary_df = pd.read_csv(args.vitals_summary)
        
        # Process labs intervals
        logger.info("Step 4: Processing labs intervals")
        labs_df = joblib.load(labs_results['processed_file'])
        intervals_dict = {
            row['Name'].strip().lower(): row['Intervals'] 
            for _, row in labs_int.iterrows()
        }
        
        labs_df['Interval Index'] = labs_df.apply(
            lambda row: assign_interval_index_vectorized(
                row['standardized_value'],
                intervals_dict.get(row['convert_sinai'].lower(), [])
            ) if row['convert_sinai'].lower() in intervals_dict else np.nan,
            axis=1
        )
        
        # Process vitals intervals
        logger.info("Step 5: Processing vitals intervals")
        df = joblib.load(vitals_output)
        df = df[df['convert_sinai'] != 'not_matched']
        df_cleaned = clean_vitals(df, summary_df)
        
        vitals_intervals_dict = {
            row['Name'].strip().lower(): row['Intervals'] 
            for _, row in vitals_int.iterrows()
        }
        
        df_cleaned['Interval Index'] = df_cleaned.apply(
            lambda row: assign_interval_index_vectorized_vitals(
                row['clean_val'], 
                vitals_intervals_dict.get(row['convert_sinai'].lower(), [])
            ) if row['convert_sinai'].lower() in vitals_intervals_dict else np.nan,
            axis=1
        )
        
        # Save results
        output_vitals = os.path.join(args.output_dir, 'vitals_intervals_cleaned.bz2')
        output_labs = os.path.join(args.output_dir, 'labs_intervals_cleaned.bz2')
        joblib.dump(df_cleaned, output_vitals, compress=('bz2',3))
        joblib.dump(labs_df, output_labs, compress=('bz2',3))
        
        logger.info("Data standardization pipeline completed successfully")
        logger.info(f"Processed files saved to: {args.output_dir}")
        logger.info(f"Lab processing reports available in: {labs_results['report_dir']}")
        logger.info(f"Final vitals file saved to: {output_vitals}")
        logger.info(f"Final labs file saved to: {output_labs}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise 

######## Example Usage ########
# python standardize_data.py \
#    --vitals_file /path/to/vitals.csv \
#    --vitals_conversion /path/to/vitals_conversion.csv \
#    --labs_file /path/to/labs.csv \
#    --labs_conversion /path/to/labs_conversion.csv \
#    --output_dir /path/to/output/ \
#    --log_dir /path/to/logs/ \
#    --labs_intervals /path/to/labs_kde_df_mover.pkl \
#    --vitals_intervals /path/to/vitals_kde_df_mover.pkl \
#    --vitals_summary /path/to/vitals_summary_statistics.csv
