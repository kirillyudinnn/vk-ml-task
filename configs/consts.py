from sklearn.metrics import f1_score, accuracy_score
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


DOMAIN_PATTERN = r'https://([^/]+)'
PATH_PATTERN = r'https://[^/]+/(.*)'

UNIQUE_BROWSER = ('Chrome Mobile', 'Chrome', 'Yandex Browser', 'Chrome Mobile WebView',
       'YandexSearch', 'Edge', 'Opera', 'Mobile Safari', 'Firefox',
       'MiuiBrowser', 'Samsung Internet', 'Safari', 'Opera Mobile',
       'Mobile Safari UI/WKWebView', 'Facebook', 'Chrome Mobile iOS', 'Google',
       'Mail.ru Chromium Browser', 'Instagram', 'UC Browser'
)
UNIQUE_OS = ('Android', 'Windows', 'iOS', 'Mac OS X', 'Linux', 'Tizen')


USER_AGENT_KEYS = ("browser", "browser_version", "os", "os_version")

CATBOOST_COLUMNS  = [
    'geo_id',
    'domain',
    'path',
    'browser',
    'os',
    'os_version',
    'component0',
    'component1',
    'component2',
    'component3',
    'component4',
    'component5',
    'component6',
    'component7',
    'component8',
    'component9',
    'country_id',
]

COLUMNS_TO_LOAD = [
    'geo_id',
    'domain',
    'path',
    'browser',
    'os',
    'os_version',
    'component0',
    'component1',
    'component2',
    'component3',
    'component4',
    'component5',
    'component6',
    'component7',
    'component8',
    'component9',
    'country_id',
    'target'
]


HSPACES = {
    "catboost" : {
            'iterations': {'type': 'int', 'low': 50, 'high': 300},
            'depth': {'type': 'int', 'low': 4, 'high': 12},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'l2_leaf_reg': {'type': 'float', 'low': 0.01, 'high': 3},
            },

    "linear" : {
            'penalty' : {'type' : 'categorical', 'values' : ['l2', 'l1']},
            'C' : {'type' : 'float', 'low' : 0.001, 'high' : 2.0},
            }
}


SCORES_DICT = {
    "accuracy" : accuracy_score,
    "f1_score" : f1_score
}

MODELS_DICT = {
    "catboost" : CatBoostClassifier,
    "linear" : LogisticRegression,
}

COLUMNS_DICT = {
    "catboost" : CATBOOST_COLUMNS,
}