import pandas
import numpy as np
from sklearn import neighbors, base

import copy
import argparse
import logging
import sys
import os
from datetime import datetime



def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_file_exists(filepath):
    """Validate that a file exists"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

def main():

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process claims data and train classifier')
    parser.add_argument('--training-data', 
                       default='./data/export_all_claims_202505211438.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--prediction-data',
                       required=True,
                       help='Path to data for predictions CSV file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Validate input files exist
        logger.info("Validating input files...")
        validate_file_exists(args.training_data)
        validate_file_exists(args.prediction_data)
        logger.info("✓ All input files found")
        
        # Load and process training data
        logger.info(f"Loading training data from: {args.training_data}")
        df = pandas.read_csv(args.training_data,
            dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
        
        logger.info(f"Initial training data shape: {df.shape}")
        
        # Filter out 'U' verdicts
        df = df[df.verdict != 'U']
        logger.info(f"After filtering 'U' verdicts: {df.shape}")
        
        # Convert verdict to binary
        df.verdict = np.array(df.verdict == 'Y', dtype=int)
        logger.info(f"Verdict distribution: {df.verdict.value_counts().to_dict()}")
        
        # Create claim feature
        df['claim'] = df.claim_origin + df.claim_type
        df = df[[
            'views',
            'matching_duration',
            'longest_match',
            'video_duration_sec',
            'verdict',
            'claim'
        ]]
        
        # One-hot encode claim types
        claim_kind = df.claim.unique()
        logger.info(f"Found {len(claim_kind)} unique claim types")
        
        for s in claim_kind:
            df[s] = np.array(df.claim == s, dtype=int)
        df = df.drop(columns='claim')
        df = df.fillna(0)
        
        # Save processed training data
        logger.info("Saving processed training data to data/YT.csv")
        df.to_csv('data/YT.csv', index=False)
        
        # Reload processed data for training
        logger.info("Reloading processed data for model training")
        df = pandas.read_csv('data/YT.csv')
        df, y = df.drop(columns='verdict'), df.verdict
        
        # Train and validate model
        logger.info("Training KNeighborsClassifier with cross-validation...")
        soln = neighbors.KNeighborsClassifier(n_neighbors=11, p=1)
        
        accuracies = []
        for i in range(4):
            test = np.random.permutation(len(df))
            test = test[:len(df) // 4]
            test = np.array([j in test for j in range(len(df))])
            
            soln.fit(df[~test], y[~test])
            valid = soln.predict_proba(df[test])
            valid = valid[:,1]
            accuracy = sum((valid > 1/2) == y[test]) / sum(test)
            accuracies.append(accuracy)
            logger.info(f"Fold {i+1}/4 accuracy: {accuracy:.4f}")
            soln = base.clone(soln)
        
        avg_accuracy = np.mean(accuracies)
        logger.info(f"Average cross-validation accuracy: {avg_accuracy:.4f}")
        
        # Train final model on all data
        logger.info("Training final model on all training data...")
        soln.fit(df, y)
        
        # Load and process prediction data
        logger.info(f"Loading prediction data from: {args.prediction_data}")
        df = pandas.read_csv(args.prediction_data,
            dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
        
        logger.info(f"Prediction data shape: {df.shape}")
        
        df2 = copy.copy(df)
        df2['claim'] = df2.claim_origin + df2.claim_type
        df2 = df2[[
            'views',
            'matching_duration',
            'longest_match',
            'video_duration_sec',
            'claim'
        ]]
        
        # Apply same one-hot encoding
        for s in claim_kind:
            df2[s] = np.array(df2.claim == s, dtype=int)
        df2 = df2.drop(columns='claim')
        df2 = df2.fillna(0)
        
        # Make predictions
        logger.info("Making predictions on new data...")
        valid = soln.predict_proba(df2)
        valid = valid[:,1]
        df['rating'] = valid
        
        # Save results
        output_file = f'data/export_unprocessed_claims_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
        logger.info(f"Saving predictions to: {output_file}")
        df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Pipeline completed successfully!")
        logger.info(f"✓ Predictions saved to: {output_file}")
        logger.info(f"✓ Rating statistics: min={valid.min():.4f}, max={valid.max():.4f}, mean={valid.mean():.4f}")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
