import os
import pandas as pd
from pathlib import Path
import warnings
import logging
import glob
import re
import sys
import argparse
from joblib import Parallel, delayed
import time
from datetime import datetime
import psutil
import threading

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import get_config, get_dataset_config, list_available_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)

logging.captureWarnings(True)
warnings_logger = logging.getLogger('py.warnings')

config = get_config()

BASE_PATH = Path(__file__).parent.parent  
SYNTHETIC_PATH_BASE = BASE_PATH / "data" / "synthetic"

# These will be set dynamically based on dataset
PROCESSED_PATH = None
RESULTS_PATH = None
TABLES_PATH = None
FIGURES_PATH = None
SYNTHETIC_PATH = None
TARGET_FEATURE = None

RANDOM_STATE = config.experiment.random_state
GENERATORS_TO_TEST = config.models.generators

SYNTHETIC_PATH_BASE.mkdir(exist_ok=True)

monitoring_active = False

def format_bytes(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}TB"


def monitor_resources(interval=None):
    global monitoring_active
    if interval is None:
        interval = config.execution.resource_monitor.interval

    while monitoring_active:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        print(f"\r CPU: {cpu_percent:5.1f}% | RAM: {format_bytes(memory.used)}/{format_bytes(memory.total)} ({memory.percent:.1f}%)", end='', flush=True)
        time.sleep(interval)


def extract_dataset_info(filename: str):
    parts = filename.replace('.csv', '').split('_')
    
    info = {
        'dataset_type': None,
        'imbalance_ratio': None,
        'repetition_id': 1  
    }
    
    if 'imbalanced' in parts:
        info['dataset_type'] = 'imbalanced'
    elif 'control' in parts:
        info['dataset_type'] = 'control'
    
    if 'ir' in parts:
        ir_idx = parts.index('ir')
        if ir_idx + 1 < len(parts):
            ir_value = re.search(r'\d+', parts[ir_idx + 1])
            if ir_value:
                info['imbalance_ratio'] = int(ir_value.group())
    
    for part in parts:
        if part.startswith('rep'):
            rep_match = re.search(r'rep(\d+)', part)
            if rep_match:
                info['repetition_id'] = int(rep_match.group(1))
    
    return info


def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def train_and_evaluate_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.drop(columns=[TARGET_FEATURE])
    y_train = train_df[TARGET_FEATURE]
    X_test = test_df.drop(columns=[TARGET_FEATURE])
    y_test = test_df[TARGET_FEATURE]

    model = RandomForestClassifier(
        random_state=config.models.random_forest.random_state,
        n_jobs=config.models.random_forest.n_jobs
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    pos_label_idx = 1 if 1 in model.classes_ else 0

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    minority_label_str = "1"

    metrics = {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_minority": report[minority_label_str]["f1-score"],
        "precision_minority": report[minority_label_str]["precision"],
        "recall_minority": report[minority_label_str]["recall"],
        "roc_auc": roc_auc_score(y_test, y_prob[:, pos_label_idx])
    }
    return metrics


def run_single_experiment(train_path: Path, test_path: Path, generator_name: str, strategy: str):
    train_df, test_df = load_data(train_path, test_path)
    dataset_info = extract_dataset_info(train_path.name)
    k_neighbors = None

    if generator_name.lower() == 'baseline':
        results = train_and_evaluate_classifier(train_df, test_df)

    elif generator_name.lower() == 'smote':
        X_train = train_df.drop(columns=[TARGET_FEATURE])
        y_train = train_df[TARGET_FEATURE]

        n_original = X_train.shape[0]

        class_counts = y_train.value_counts()
        n_minority = int(class_counts.min())
        k_neighbors = min(5, n_minority - 1)

        total_size = len(train_df)
        n_per_class = total_size // 2

        ## to replace completely the original data and have the same size as control
        sampling_strategy = {
            int(cls): int(class_counts[cls]) + n_per_class
            for cls in class_counts.index
        }

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=k_neighbors
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)

        assert pd.DataFrame(X_res[:n_original], columns=X_train.columns).equals(X_train.reset_index(drop=True)), "Originals were modified!"

        X_resampled = X_res[n_original:]
        y_resampled = y_res[n_original:]

        X_orig_set = set(map(tuple, X_train.values))
        duplicates = sum(1 for row in X_resampled if tuple(row) in X_orig_set)
        if duplicates > 0:
            print(f"[SMOTE] Warning!! {duplicates} synthetic points exactly match original rows.")

        balanced_train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
        balanced_train_df[TARGET_FEATURE] = y_resampled

        synth_filename = f"{train_path.stem}_synthetic_{generator_name}.csv"
        balanced_train_df.to_csv(SYNTHETIC_PATH / synth_filename, index=False)

        results = train_and_evaluate_classifier(balanced_train_df, test_df)

    else:
        loader = GenericDataLoader(train_df, target_column=TARGET_FEATURE)
        generator = Plugins().get(generator_name)
        generator.fit(loader, cond=train_df[TARGET_FEATURE])

        value_counts = train_df[TARGET_FEATURE].value_counts()
        total_samples = len(train_df)
        class_labels = value_counts.index.tolist()

        # Create balanced conditions (50% each class)
        n_per_class = total_samples // 2
        conditions = []
        for label in class_labels:
            conditions.extend([label] * n_per_class)

        conditions = pd.Series(conditions, name=TARGET_FEATURE)

        synth_data = generator.generate(count=len(conditions), cond=conditions).dataframe()

        synth_filename = f"{train_path.stem}_synthetic_{generator_name}.csv"
        synth_data.to_csv(SYNTHETIC_PATH / synth_filename, index=False)

        balanced_train_df = synth_data

        results = train_and_evaluate_classifier(balanced_train_df, test_df)

    result_log = {
        "repetition_id": dataset_info['repetition_id'],
        "dataset_type": dataset_info['dataset_type'],
        "imbalance_ratio": f"{dataset_info['imbalance_ratio']}:1",
        "model": generator_name,
        "strategy": strategy,
        "train_set_size": len(train_df),
        "k_neighbors": k_neighbors if generator_name.lower() == 'smote' else None,
        **results 
    }
    return result_log


def process_single_dataset(train_path: Path, test_path: Path, dataset_idx: int, total_datasets: int):
    results = []
    dataset_info = extract_dataset_info(train_path.name)
    start_time = time.time()

    logger.info(f"[{dataset_idx}/{total_datasets}] Processing: {train_path.name}")

    # Baseline
    try:
        logger.info(f"  → Running baseline...")
        baseline_result = run_single_experiment(train_path, test_path, "baseline", "N/A")
        results.append(baseline_result)
        logger.info(f"  ✓ Baseline completed")
    except Exception as e:
        logger.error(f"Baseline failed on {train_path.name}: {e}", exc_info=True)

    # Generators
    for generator in GENERATORS_TO_TEST:
        try:
            logger.info(f"  → Running {generator}...")
            synthetic_result = run_single_experiment(
                train_path, test_path, generator, "Fully-Synthetic Balanced"
            )
            results.append(synthetic_result)
            logger.info(f"  ✓ {generator} completed")
        except Exception as e:
            logger.error(f"Generator '{generator}' failed on {train_path.name}: {e}", exc_info=True)

    elapsed = time.time() - start_time
    logger.info(f"[{dataset_idx}/{total_datasets}] Completed in {elapsed:.1f}s")

    return results

class ProgressTracker:
    def __init__(self, total_datasets):
        self.total_datasets = total_datasets
        self.completed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def increment(self):
        with self.lock:
            self.completed += 1
            self._display_progress()
    
    def _display_progress(self):
        elapsed = time.time() - self.start_time
        if self.completed > 0:
            avg_time = elapsed / self.completed
            remaining = avg_time * (self.total_datasets - self.completed)
            eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
        else:
            eta = "calculating..."
        
        percent = (self.completed / self.total_datasets) * 100
        bar_length = 50
        filled = int(bar_length * self.completed / self.total_datasets)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\rProgress: [{bar}] {percent:5.1f}% ({self.completed}/{self.total_datasets}) | ETA: {eta}   ", end='', flush=True)


def setup_dataset_paths(dataset_name: str, timestamp: str):
    global PROCESSED_PATH, RESULTS_PATH, TABLES_PATH, FIGURES_PATH, SYNTHETIC_PATH, TARGET_FEATURE

    # Load dataset configuration
    dataset_config = get_dataset_config(dataset_name)

    # Set paths based on dataset configuration
    PROCESSED_PATH = BASE_PATH / dataset_config['processed_path']
    RESULTS_PATH = BASE_PATH / dataset_config['results_path']
    TABLES_PATH = RESULTS_PATH / "tables"
    FIGURES_PATH = RESULTS_PATH / "figures"
    SYNTHETIC_PATH = SYNTHETIC_PATH_BASE / dataset_name / f"run_{timestamp}"
    TARGET_FEATURE = dataset_config['target_column']

    # Create directories
    SYNTHETIC_PATH.mkdir(parents=True, exist_ok=True)
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def main(dataset_name: str = "mammographic_mass"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup paths for this dataset
    setup_dataset_paths(dataset_name, timestamp)

    logger.info(f"Experimental Pipeline - Dataset: {dataset_name}")

    logger.info(f"Dataset Configuration:")
    logger.info(f"  • Target column: {TARGET_FEATURE}")
    logger.info(f"  • Processed data: {PROCESSED_PATH}")
    logger.info(f"  • Results path: {RESULTS_PATH}")
    logger.info(f"  • Synthetic data: {SYNTHETIC_PATH}")

    start_time = time.time()
    test_path = PROCESSED_PATH / "test.csv"

    train_paths = sorted(glob.glob(str(PROCESSED_PATH / "train_*.csv")))
    
    if not train_paths:
        logger.error(f"No training datasets found in {PROCESSED_PATH}")
        return
    
    n_datasets = len(train_paths)
    n_methods = len(GENERATORS_TO_TEST) + 1
    total_experiments = n_datasets * n_methods
    
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    
    logger.info(f"System Information:")
    logger.info(f"   • CPU Cores: {cpu_count}")
    logger.info(f"   • Total RAM: {format_bytes(memory.total)}")
    logger.info(f"   • Available RAM: {format_bytes(memory.available)}")
    
    logger.info(f"Experiment Configuration:")
    logger.info(f"   • Datasets: {n_datasets}")
    logger.info(f"   • Methods: {n_methods} (1 baseline + {len(GENERATORS_TO_TEST)} generators)")
    logger.info(f"   • Total experiments: {total_experiments}")
    logger.info(f"   • Parallel workers: {cpu_count}")
    
    global monitoring_active
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    train_paths = [Path(p) for p in train_paths]
    
    try:
        all_results_nested = Parallel(
            n_jobs=config.execution.parallel.n_jobs,
            backend=config.execution.parallel.backend,
            verbose=10
        )(
            delayed(process_single_dataset)(train_path, test_path, idx+1, n_datasets)
            for idx, train_path in enumerate(train_paths)
        )
    finally:
        monitoring_active = False
        time.sleep(0.5) 
        print("\r" + " "*120)  
    
    all_results = [result for dataset_results in all_results_nested for result in dataset_results]
    
    elapsed_time = time.time() - start_time
    
    logger.info("All experiments complete!")
    logger.info(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    logger.info(f"Speed: {total_experiments/elapsed_time:.2f} experiments/second")
    
    results_df = pd.DataFrame(all_results)

    summary_df = results_df.groupby(['dataset_type', 'imbalance_ratio', 'model', 'strategy']).agg(
        {
            'f1_minority': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'recall_minority': ['mean', 'std'],
            'precision_minority': ['mean', 'std']
        }
    ).reset_index()

    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.rename(columns={
        'dataset_type_': 'dataset_type',
        'imbalance_ratio_': 'imbalance_ratio',
        'model_': 'model',
        'strategy_': 'strategy'
    })

    run_folder = TABLES_PATH / f"run_{timestamp}"
    run_folder.mkdir(exist_ok=True)

    results_df.to_csv(run_folder / "detailed_experiment_results.csv", index=False)
    summary_df.to_csv(run_folder / "summary_experiment_results.csv", index=False)

    logger.info(f"Results saved to: {run_folder}")
    logger.info(f"Synthetic data saved to: {SYNTHETIC_PATH}")
    
    logger.info("Performance by Model (F1-Minority / ROC-AUC):")
    
    summary_table = results_df.groupby('model').agg({
        'f1_minority': ['mean', 'std'],
        'roc_auc': ['mean', 'std']
    }).round(4)
    
    for model in summary_table.index:
        f1_mean = summary_table.loc[model, ('f1_minority', 'mean')]
        f1_std = summary_table.loc[model, ('f1_minority', 'std')]
        auc_mean = summary_table.loc[model, ('roc_auc', 'mean')]
        auc_std = summary_table.loc[model, ('roc_auc', 'std')]
        logger.info(f"   {model:<15} | F1: {f1_mean:.4f} (±{f1_std:.4f}) | AUC: {auc_mean:.4f} (±{auc_std:.4f})")
            
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run synthetic data generation experiments on tabular datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run experiments on a single dataset
            python run_experiment.py --dataset mammographic_mass

            # Run experiments on all registered datasets
            python run_experiment.py --dataset all

            # List available datasets
            python run_experiment.py --list-datasets
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mammographic_mass",
        help="Name of the dataset to run experiments on, or 'all' to run on all datasets"
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # List datasets if requested
    if args.list_datasets:
        print("Available datasets:")
        for dataset in list_available_datasets():
            dataset_config = get_dataset_config(dataset)
            print(f"  • {dataset}")
            print(f"    - Target: {dataset_config['target_column']}")
            print(f"    - Path: {dataset_config['processed_path']}")
            if 'description' in dataset_config:
                print(f"    - Description: {dataset_config['description']}")
        sys.exit(0)

    # Run experiments on specified dataset(s)
    if args.dataset.lower() == "all":
        logger.info("Running experiments on all datasets...")
        available_datasets = list_available_datasets()
        logger.info(f"Found {len(available_datasets)} datasets: {', '.join(available_datasets)}")

        for dataset in available_datasets:
            try:
                logger.info(f"Starting experiments for dataset: {dataset}")
                main(dataset)
            except Exception as e:
                logger.error(f"Error running experiments for {dataset}: {e}", exc_info=True)
                continue

        logger.info("All dataset experiments completed!")
    else:
        # Run on single dataset
        try:
            main(args.dataset)
        except ValueError as e:
            logger.error(f"{e}")
            print("\nUse --list-datasets to see available datasets")
            sys.exit(1)
        except Exception as e:
            logger.error(f"{e}", exc_info=True)
            sys.exit(1)