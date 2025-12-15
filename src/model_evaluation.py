import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
import yaml
from sklearn.metrics import r2_score
from dvclive import Live

# ---------------- LOGGING ----------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model-evaluation')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(log_dir, 'model_evaluation.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------- FUNCTIONS ----------------
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Failed to load params: %s', e)
        raise


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logger.error('Failed to load model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        logger.debug('Model evaluation completed. R2: %.4f', r2)
        return {"r2_score": r2}
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Failed to save metrics: %s', e)
        raise

def close_logger():
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

# ---------------- MAIN ----------------
def main():
    try:
        params = load_params('params.yaml')

        model = load_model('./models/model.pkl')
        test_data = load_data('data/interim/test_processed.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)

        # ---- CLOSE LOG FILE BEFORE DVC ----
        close_logger()

        with Live(save_dvc_exp=True) as live:
            for k, v in metrics.items():
                live.log_metric(k, v)

        save_metrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error('Model evaluation pipeline failed: %s', e)
        raise


if __name__ == '__main__':
    main()
