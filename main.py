import typer
from structlog import get_logger
from datetime import datetime

import numpy as np
from sklearn.metrics import classification_report

from configs.consts import (
    DOMAIN_PATTERN, 
    PATH_PATTERN,
    SCORES_DICT,
    MODELS_DICT,
    HSPACES,
    CATBOOST_COLUMNS,
    COLUMNS_TO_LOAD
)
from utils.utils import (
    split_data_by_user_id, 
    write_data_on_storage, 
    load_data, 
    generate_features_from_columns,
    prune_os_browser_categories,
)

from trainer.optuna_trainer import HyperparameterTrainer

app = typer.Typer()


@app.command()
def load_and_preprocess_data(
    train_size: float = typer.Option(..., envvar="TRAIN_SIZE"),
    val_size: float = typer.Option(..., envvar="VAL_SIZE"),
    test_size: float = typer.Option(..., envvar="TEST_SIZE"),
    execution_date: datetime = typer.Option(..., envvar="EXECUTION_DATE"),
    mode: str = typer.Option(..., envvar="MODE")
):
    logger = get_logger()
    logger.info("-----------------------[LOAD AND PREPROCESS DATA TASK]--------------------------")
    date = execution_date.strftime("%Y-%m-%d")
    PREPROCESSED_DATA_PATH = f"./data/{date}/"
    PATH_TO_LOCAL_DATA = r'./data/'

    assert np.isclose(train_size + val_size + test_size, 1.0), "Сумма долей должна быть равна 1"

    logger.info("Load data from source")

    train_labels = load_data(path_to_data=PATH_TO_LOCAL_DATA + "train_labels.csv", sep=";")
    referer_vectors = load_data(path_to_data=PATH_TO_LOCAL_DATA + "referer_vectors.csv", sep=";")
    geo_info = load_data(path_to_data=PATH_TO_LOCAL_DATA + "geo_info.csv", sep=";")
    train = load_data(path_to_data=PATH_TO_LOCAL_DATA + "train.csv", sep=";")

    test = load_data(path_to_data=PATH_TO_LOCAL_DATA + "test.csv", sep=";")

    logger.info("Generate features from columns")
    train = generate_features_from_columns(train)
    test = generate_features_from_columns(test)

    logger.info("Filtering os and browser columns. Replace rare values for 'other'")
    train = prune_os_browser_categories(train)
    test = prune_os_browser_categories(test)


    logger.info("Preprocessing referer data")
    referer_vectors['domain'] = referer_vectors['referer'].str.extract(DOMAIN_PATTERN)
    referer_vectors['path'] = referer_vectors['referer'].str.extract(PATH_PATTERN)


    logger.info("Split unlabeled users data")
    users_with_label = set(train_labels['user_id'].values)
    known_users_mask = train['user_id'].isin(users_with_label)

    labeled_train = train[known_users_mask].copy()
    del train

    referer_vectors_dupl_index = referer_vectors[referer_vectors.duplicated()].index
    referer_vectors = referer_vectors.drop(referer_vectors_dupl_index)
    referer_vectors.drop("referer", axis=1, inplace=True)

    df_train = labeled_train.merge(
        referer_vectors,
        on=['domain', 'path'],
        how='left'
    )

    df_train = df_train.merge(
        geo_info,
        on=['geo_id'],
        how='left'
    )

    df_train = df_train.merge(
        train_labels,
        on='user_id',
        how='left'
    )

    test = test.merge(
        referer_vectors,
        on=['domain', 'path'],
        how='left'
    )

    test = test.merge(
        geo_info,
        on=['geo_id'],
        how='left'
    )

    logger.info(f'''
        Split dataset by user_id
        Training size: {train_size}
        Val size: {val_size}
        Test size: {test_size}
            '''
    )

    train_ids, val_ids, test_ids = split_data_by_user_id(
        user_ids = df_train['user_id'].unique(), 
        train_size = train_size, 
        val_size = val_size, 
    )

    Xy_train = df_train[df_train['user_id'].isin(train_ids)].copy()
    Xy_val = df_train[df_train['user_id'].isin(val_ids)].copy()
    Xy_test = df_train[df_train['user_id'].isin(test_ids)].copy()

    logger.info("Writing data")
    columns_to_drop = ["request_ts"]
    write_data_on_storage(Xy_train, PREPROCESSED_DATA_PATH, "Xy_train.csv", columns_to_drop)
    write_data_on_storage(Xy_val, PREPROCESSED_DATA_PATH, "Xy_val.csv", columns_to_drop)
    write_data_on_storage(Xy_test, PREPROCESSED_DATA_PATH, "Xy_test.csv", columns_to_drop)

    logger.info("Done!")


@app.command()
def train_and_evaluate_model(
    model_name: str = typer.Option(..., envvar="MODEL_NAME"),
    n_trials: int = typer.Option(..., envvar="N_TRIALS"),
    scoring: str = typer.Option(..., envvar="SCORING"),
    execution_date: datetime = typer.Option(..., envvar="EXECUTION_DATE"),
):
    date = execution_date.strftime("%Y-%m-%d")
    PREPROCESSED_DATA_PATH = f"./data/{date}/"
    scoring_func = SCORES_DICT[scoring]
    model_class = MODELS_DICT[model_name]
    hspace = HSPACES[model_name]

    if model_name == "catboost":
        additional_model_params = {"cat_features" : CATBOOST_COLUMNS, "verbose" : 50}
        early_stopping = True
    else:
        additional_model_params = {}
        early_stopping = False

    logger = get_logger()
    logger.info("--------------------------[TRAINING TASK]--------------------------")
    logger.info(f"Start training {model_name} with {n_trials} Optuna Trials")

    df_train = load_data(PREPROCESSED_DATA_PATH + "Xy_train.csv", usecols=COLUMNS_TO_LOAD)
    df_val = load_data(PREPROCESSED_DATA_PATH + "Xy_val.csv", usecols=COLUMNS_TO_LOAD)
    df_test = load_data(PREPROCESSED_DATA_PATH + "Xy_test.csv", usecols=COLUMNS_TO_LOAD)

    X_train, y_train = df_train.drop("target", axis=1), df_train["target"].values
    X_val, y_val = df_val.drop("target", axis=1), df_val["target"].values
    X_test, y_test = df_test.drop("target", axis=1), df_test["target"].values

    X_train.fillna('', inplace=True)
    X_val.fillna('', inplace=True)
    X_test.fillna('', inplace=True)


    trainer = HyperparameterTrainer(
        model_class=model_class,
        hpspace=hspace,
        n_trials=n_trials,
        additional_model_params=additional_model_params,
        early_stopping=early_stopping
    )

    best_model = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        score_func=scoring_func,
        X_val=X_val,
        y_val=y_val
    )

    best_params, best_score = trainer.get_best_params(), trainer.get_best_score()
    logger.info(f"Best parameteres: {best_params}")
    logger.info(f"Best {scoring} score: {best_score}")

    y_pred = best_model.predict(X_test)
    clf_report = classification_report(y_test, y_pred)

    logger.info(clf_report)

    logger.info("Done!")


if __name__ == "__main__":
    app()
