#!/usr/bin/env python3
# _____pure import_______
import numpy as np
import pandas as pd
import logging
import logging.config
# _____module import_______
from sklearn import metrics, model_selection, linear_model
import custom_functions


logging.basicConfig(filename="LogReg_classifier_validation.log",
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%m-%Y %I:%M:%S',
                    level=logging.INFO)
log = logging.getLogger("LogReg_classifier_validation")
# _____data preporation________
log.info("data preprocessing")
df = pd.read_csv('forestfires.csv', header=0, sep=',')
target = df['area']
class_target = np.zeros(shape=len(target))
class_target[target > 0] = 1
drop_list = ['area', 'day']
raw_data = df.drop(drop_list, axis=1)
raw_data['month'] = [custom_functions.month_mapper(month) for month in raw_data['month']]
tr_data, t_data, tr_target, t_target = model_selection.train_test_split(raw_data, class_target, test_size=0.3,
                                                                        random_state=1)
log.info('dividing features by types')
coord_data_cols = ['X', 'Y']
coord_data_ind = np.array([(column in coord_data_cols) for column in tr_data.columns], dtype=bool)

numeric_col = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
numeric_data_ind = np.array([(column in numeric_col) for column in tr_data.columns], dtype=bool)

date_data_cols = ['month']
date_data_ind = np.array([(column in date_data_cols) for column in tr_data.columns], dtype=bool)

# _____Classification task_______
log.info('Classification task')
classifier = linear_model.LogisticRegression()
estimator = custom_functions.create_estimator(classifier, numeric_data_ind, coord_data_ind, date_data_ind)

param_grid = [
    {
        'Model_fitting__C': np.concatenate([np.linspace(10**-10, 10**-5, 200), np.linspace(10**-5, 10**5, 200)]),
        'Model_fitting__penalty': ['l2'],
        'Model_fitting__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'Model_fitting__max_iter': np.arange(5000, 50000, step=50000)
    },
    {
        'Model_fitting__penalty': ['none'],
        'Model_fitting__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'Model_fitting__max_iter': np.arange(5000, 50000, step=50000)
    },
    {
        'Model_fitting__C': np.concatenate([np.linspace(10**-10, 10**-5, 200), np.linspace(10**-5, 10**5, 200)]),
        'Model_fitting__penalty': ['l1'],
        'Model_fitting__solver': ['liblinear', 'saga'],
        'Model_fitting__max_iter': np.arange(5000, 50000, step=50000)
    },
    {
        'Model_fitting__C': np.concatenate([np.linspace(10**-10, 10**-5, 200), np.linspace(10**-5, 10**5, 200)]),
        'Model_fitting__penalty': ['elasticnet'],
        'Model_fitting__solver': ['saga'],
        'Model_fitting__max_iter': np.arange(5000, 50000, step=5000),
        'Model_fitting__l1_ratio': np.arange(0.01, 1, step=0.05)
    }
]

log.info('Validation')
scorer_1 = metrics.make_scorer(
    metrics.precision_score,
    pos_label=0
)
scorer_2 = metrics.make_scorer(
    metrics.fbeta_score,
    beta=1.3
)
scorer_3 = 'recall'
scorer_4 = 'f1'
scorers = [scorer_1, scorer_2, scorer_3, scorer_4]
for scorer in scorers:
    log.info('Validation with scorer')
    grid_cv = model_selection.GridSearchCV(estimator=estimator,
                                           param_grid=param_grid,
                                           scoring=scorer,
                                           cv=5,
                                           verbose=100,
                                           n_jobs=-1)
    grid_cv.fit(X=tr_data, y=tr_target)
    log.info('Best score: {0}'.format(grid_cv.best_score_))
    log.info("Best params {0}".format(grid_cv.best_params_))
    b_est = grid_cv.best_estimator_

    log.info('Some metrics on train set')
    pred = b_est.predict(tr_data)
    log.info(metrics.accuracy_score(tr_target, pred))
    log.info('ROC AUC score {0}'.format(metrics.roc_auc_score(tr_target, pred)))
    log.info(metrics.classification_report(tr_target, pred))

    log.info('Some metrics on test set')
    pred = b_est.predict(t_data)
    log.info('Accuracy score {0}'.format(metrics.accuracy_score(t_target, pred)))
    log.info('ROC AUC score {0}'.format(metrics.roc_auc_score(t_target, pred)))
    log.info(metrics.classification_report(t_target, pred))


    log.info('Some metrics on full dataset')
    pred = b_est.predict(raw_data)
    log.info('Accuracy score {0}'.format(metrics.accuracy_score(class_target, pred)))
    log.info('ROC AUC score {0}'.format(metrics.roc_auc_score(class_target, pred)))
    log.info(metrics.classification_report(class_target, pred))