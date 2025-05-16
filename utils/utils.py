import numpy as np
from typing import List, Tuple
import pandas as pd
from structlog import get_logger

import os
import ast

from configs.consts import (
    USER_AGENT_KEYS,
    DOMAIN_PATTERN,
    PATH_PATTERN,
    UNIQUE_OS,
    UNIQUE_BROWSER
)

logger = get_logger()


def split_data_by_user_id(
    user_ids: List[str],
    train_size: float,
    val_size: float,
    random_state: int = 21
)-> Tuple[List[str], List[str], List[str]]:
    
    rng = np.random.RandomState(random_state)
    shuffled_ids = rng.permutation(user_ids)
    
    n_total = len(user_ids)
    n_train = int(n_total * train_size)
    n_val = int(n_total * val_size)
    
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train : n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids
    

def generate_features_from_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['event_datetime'] = pd.to_datetime(df['request_ts'], unit='s')
    
    logger.info("Eval user_agent column")
    df["user_agent"] = df["user_agent"].fillna(r"{}")
    df['user_agent'] = df['user_agent'].apply(ast.literal_eval)

    logger.info(f"Get values from user_agent with keys {USER_AGENT_KEYS}")

    for key in USER_AGENT_KEYS:
        df[key] = df["user_agent"].str.get(key)

    logger.info("Parsing domain and path from referer")
    df['domain'] = df['referer'].str.extract(DOMAIN_PATTERN)
    df['path'] = df['referer'].str.extract(PATH_PATTERN)

    return df


def prune_os_browser_categories(df: pd.DataFrame):
    logger.info(f"Unique OS values: {UNIQUE_OS}")
    logger.info(f"Unique Browser values: {UNIQUE_BROWSER}")

    browser_mask = df["browser"].isin(UNIQUE_BROWSER)
    df.loc[~browser_mask, "browser"] = 'other'
    
    os_mask = df["os"].isin(UNIQUE_OS)
    df.loc[~os_mask, "os"] = 'other'

    return df

def write_data_on_storage(df: pd.DataFrame, directory_path: str, filename: str, cols_to_drop: List[str] = []):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    
    if cols_to_drop:
        df.drop(cols_to_drop, axis=1, inplace=True)

    df.to_csv(directory_path + filename, index=False)
    

def load_data(path_to_data: str, source_type: str = "local", sep: str = ',', usecols: List[str]=None):
    if source_type == "local":
        return pd.read_csv(path_to_data, sep=sep, usecols=usecols)
    

def split_X_y(df: pd.DataFrame, target_name: str = "target"):
    y = df[target_name].copy()
    return df.drop(target_name, axis=1, inplace=True), y