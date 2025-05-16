import optuna
from optuna.samplers import TPESampler
from typing import Dict, Any

class HyperparameterTrainer:
    def __init__(
        self,
        model_class,
        hpspace: Dict[str, Any],
        n_trials: int = 10,
        additional_model_params: Dict[str, Any] = {},
        random_state: int = 21,
        early_stopping = False
    ):
        """
        Trainer for tuning hyperparameters 
        
        Args:
            model_class: Model class (CatBoostClassifier or LogisticRegression)
            hpspace: hyperparameters space
            n_trials: N tuning trials
            additional_model_params
            scoring: Optimization metric
            random_state: random state
        """
        self.model_class = model_class
        self.hpspace = hpspace
        self.n_trials = n_trials
        self.additional_model_params = additional_model_params
        self.random_state = random_state
        self.study = None
        self.best_model = None
        self.early_stoppping = early_stopping

    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = {}
        for param_name, param_config in self.hpspace.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['values'])
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high'], 
                    log=param_config.get('log', False))
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], 
                    log=param_config.get('log', False))
        
        model = self.model_class(**params, **self.additional_model_params)
        if self.early_stoppping:
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = self.scoring(y_val, y_pred)
        return score

    def fit(self, X_train, y_train, score_func, direction="maximize", X_val=None, y_val=None):
        if self.early_stoppping and (X_val is None or y_val is None):
            raise ValueError("Eval set is required if early_stopping is True")
        
        self.scoring = score_func
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction=direction, sampler=sampler)
        
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials
        )
        
        self.best_model = self.model_class(
            **self.study.best_params, 
            **self.additional_model_params
        )
        if self.early_stoppping:
            self.best_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        return self.best_model

    def get_best_params(self):
        return self.study.best_params if self.study else None

    def get_best_score(self):
        return self.study.best_value if self.study else None