#!/usr/bin/env python3
# _____pure import_______
import numpy as np
import pandas as pd
import logging
import logging.config
# _____module import_______
from sklearn import metrics, model_selection, ensemble

import custom_functions

logging.basicConfig(filename="RF_classifier_validation.log",
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m%d%Y %I:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("RF_classifier_validation")


# _____data preporation________
logger.info("RF: data preprocessing")
df = pd.read_csv('forestfires.csv', header=0, sep=',')
target = df['area']
class_target = np.zeros(shape=len(target))
class_target[target > 0] = 1
drop_list = ['area', 'day']
raw_data = df.drop(drop_list, axis=1)
raw_data['month'] = [custom_functions.month_mapper(month) for month in raw_data['month']]
tr_data, t_data, tr_target, t_target = model_selection.train_test_split(raw_data, class_target, test_size=0.3,
                                                                        random_state=1)
logger.info('RF: dividing features by types')
coord_data_cols = ['X', 'Y']
coord_data_ind = np.array([(column in coord_data_cols) for column in tr_data.columns], dtype=bool)

numeric_col = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
numeric_data_ind = np.array([(column in numeric_col) for column in tr_data.columns], dtype=bool)

date_data_cols = ['month']
date_data_ind = np.array([(column in date_data_cols) for column in tr_data.columns], dtype=bool)

# _____Classification task_______
logger.info('RF: Classification task')
classifier = ensemble.RandomForestClassifier()
estimator = custom_functions.create_estimator(classifier, numeric_data_ind, coord_data_ind, date_data_ind)

param_grid = {
    'Model_fitting__bootstrap': [True],
    'Model_fitting__n_estimators': np.arange(20, 100, step=2),
    'Model_fitting__criterion': ['gini', "entropy"],
    'Model_fitting__min_samples_leaf': np.arange(1, 20, step=1),
    'Model_fitting__min_samples_split': np.arange(2, 20, step=1),
    'Model_fitting__max_features': ['sqrt', 'log2', None],
    'Model_fitting__oob_score': [True, False],
    'Model_fitting__random_state': [1]
}

logger.info('Validation')
grid_cv = model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='f1', cv=5)
grid_cv.fit(X=tr_data, y=tr_target)
logger.info('Best score: {0}'.format(grid_cv.best_score_))
print("Best params {0}".format(grid_cv.best_params_))
b_est = grid_cv.best_estimator_
pred = b_est.predict(t_data)

logger.info('Some metrics on test set')
logger.info(metrics.accuracy_score(t_target, pred))
logger.info(metrics.classification_report(t_target, pred))

logger.info("Some metrics in full data")
estimator.fit(X=raw_data, y=class_target)
pred = estimator.predict(raw_data)
print(metrics.accuracy_score(class_target, pred))
print(metrics.classification_report(class_target, pred))
