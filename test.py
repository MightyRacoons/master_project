#!/usr/bin/env python3
#_____pure import_______
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#_____module import_______
from scipy import stats
from matplotlib.colors import ListedColormap
from sklearn import metrics, preprocessing, model_selection, tree, decomposition, ensemble, pipeline, linear_model, svm
from sklearn.pipeline import Pipeline

#_____custom functions_____
def plot_2d_data(data, target, colors):
    plt.figure(figsize = [8,8])
    x = data[0]
    plt.scatter(x[:,0], x[:,1],c = target, cmap = colors)
    plt.show()

def plot_d_surface(estimator, tr_data, tr_labels, t_data, t_labels,colors, light_colors):
    estimator.fit(tr_data, tr_labels)
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    xx,yy = np.meshgrid(tr_data[:,0], tr_data[:,1])
    mesh_pred = np.array(estimator.predict(np.c_[xx.ravel(),yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx,yy, mesh_pred, cmap=light_colors)
    plt.scatter(tr_data[:,0], tr_data[:,1], c = tr_labels, cmap = colors)
    plt.title('Train data, accuracy={:.2f}'.format(metrics.accuracy_score(tr_labels, estimator.predict(tr_data))))
    plt.subplot(1,2,2)
    plt.pcolormesh(xx,yy, mesh_pred, cmap=light_colors)
    plt.scatter(t_data[:,0], t_data[:,1], c = t_labels, cmap = colors)
    plt.title('Test data, accuracy={:.2f}'.format(metrics.accuracy_score(t_labels,estimator.predict(t_data))))
    plt.show()


def month_mapper(month):
    m = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr':4,
        'may':5,
        'jun':6,
        'jul':7,
        'aug':8,
        'sep':9,
        'oct':10,
        'nov':11,
        'dec':12}
    s = month.strip()[:3].lower()
    return m[s]

def day_mapper(day):
    d ={
        'mon':1,
        'tue':2,
        'wed':3,
        'thu':4,
        'fri':5,
        'sat':6,
        'sun':7
    }
    s = day.strip()[:3].lower()
    return d[s]

def createEstimator(ml_obj):
    estimator = Pipeline(steps=[
        ('Feature_processing',pipeline.FeatureUnion(transformer_list=[
            ('Numeric_features', Pipeline(steps=[
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:,numeric_data_ind], validate=True)),
                ('scaling', preprocessing.StandardScaler(with_mean=0., with_std=1))
            ])),
            ('Categical_features', Pipeline(steps=[
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:,coord_data_ind], validate=True )),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore'))
            ])),
            ('Date_features', Pipeline(steps=[
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:,date_data_ind], validate=True )),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore'))
            ]))
        ])),
        ('Model_fitting', ml_obj)
    ])
    return estimator


#_____data preporation________
df = pd.read_csv('forestfires.csv', header = 0, sep=',')
target = df['area']
class_target = np.zeros(shape= len(target))
class_target[target > 0] = 1
drop_list = ['area', 'day']
raw_data = df.drop(drop_list, axis=1)
raw_data['month'] = [month_mapper(month) for month in raw_data['month']]
tr_data, t_data, tr_target, t_target = model_selection.train_test_split(raw_data , class_target, test_size = 0.3, random_state = 1)

coord_data_cols = ['X','Y']
coord_data_ind = np.array([(column in coord_data_cols) for column in tr_data.columns], dtype=bool)

numeric_col = ['FFMC','DMC','DC','ISI','temp','RH','wind','rain']
numeric_data_ind = np.array([(column in numeric_col) for column in tr_data.columns], dtype=bool)

date_data_cols = ['month']
date_data_ind = np.array([(column in date_data_cols) for column in tr_data.columns], dtype = bool)

#_____Classification task_______
#classifier = tree.DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=9, min_samples_leaf=5, min_samples_split=16, splitter="random")
classifier = ensemble.RandomForestClassifier(n_jobs=-1, bootstrap=True, criterion="entropy",
                                             max_features="sqrt", min_samples_leaf=1, min_samples_split=2,
                                             n_estimators=88, oob_score=True, random_state=1, max_depth = 5)
#classifier = linear_model.LogisticRegression(random_state=1, C=0.5, penalty='l2', max_iter=20, solver='lbfgs')
#classifier = svm.SVC(random_state=1, C=0.75, kernel='poly', degree=2, coef0 = 3.5)
#classifier = ensemble.GradientBoostingClassifier(random_state=1)
estimator = createEstimator(classifier)
# print(classifier.get_params(deep=True).keys())
# print(estimator.get_params().keys())
# param_grid = {
#     'Model_fitting__criterion':['gini',"entropy" ],
#     'Model_fitting__splitter': ['best','random'],
#     'Model_fitting__max_depth':np.arange(start=1, stop=10, step=1),
#     'Model_fitting__min_samples_split':np.arange(start=2, stop=20, step=1),
#     'Model_fitting__min_samples_leaf': np.arange(start=1, stop = 10, step = 1),
#     'Model_fitting__random_state': [1]
# }
# param_grid = {
#     'Model_fitting__bootstrap':[True],
#     'Model_fitting__n_estimators':np.arange(20,100,step = 2),
#     'Model_fitting__criterion':['gini',"entropy" ],
#     'Model_fitting__min_samples_leaf':np.arange(1,20,step =1),
#     'Model_fitting__min_samples_split': np.arange(2,20,step = 1),
#     'Model_fitting__max_features':['sqrt','log2',None],
#     'Model_fitting__oob_score':[True, False],
#     'Model_fitting__random_state':[1]
# }
# param_grid = {
#     'Model_fitting__C': np.arange(0.1,1.0, step = 0.05),
#     'Model_fitting__penalty': ['l2'],
#     'Model_fitting__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#     'Model_fitting__max_iter': np.arange(10, 1000, step = 10)
# }
# param_grid = {
#     'Model_fitting__C': np.arange(0.1, 1.0, step=0.05),
#     'Model_fitting__kernel': [ 'linear', 'poly', 'rbf', 'sigmoid'],
#     'Model_fitting__degree': np.arange(start=2, stop=6, step=1),
#     'Model_fitting__coef0': np.arange(start=0, stop=10, step=0.5)
# }
# param_grid = {
#     'Model_fitting__loss':['deviance', 'exponential'],
#     'Model_fitting__learning_rate':np.arange(0.05, 1, step = 0.05),
#     'Model_fitting__n_estimators':np.arange(20,100,step = 2),
#     'Model_fitting__min_samples_leaf':np.arange(1,20,step =1),
#     'Model_fitting__min_samples_split': np.arange(2,20,step = 1),
#     'Model_fitting__subsample': np.arange(0.1, 0.9, step = 0.1),
#     'Model_fitting__max_features':['sqrt','log2',None]
# }
#
#
# grid_cv = model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy', cv=5)
# grid_cv.fit(X=tr_data, y=tr_target)
# print(grid_cv.best_score_)
# print(grid_cv.best_params_)
# b_est = grid_cv.best_estimator_
# pred = b_est.predict(t_data)
#

# estimator.fit(X= tr_data,y = tr_target)
# pred = estimator.predict(t_data)
# print(metrics.accuracy_score(t_target, pred))
# print(metrics.classification_report(t_target, pred))
#

estimator.fit(X=raw_data, y=class_target)
pred =estimator.predict(raw_data)
print(metrics.accuracy_score(class_target, pred))
print(metrics.classification_report(class_target, pred))



#______Regression task_______
regr_data = raw_data.iloc[pred==1]
target = target[pred==1]
target = np.log(target+1)
# print("______Regression task_______")
# regressor = ensemble.GradientBoostingRegressor()
# estimator = createEstimator(regressor)
# print(estimator.get_params().keys())
# param_grid = {
#     'Model_fitting__loss':np.array(['ls','lad','huber','quantile']),
#     'Model_fitting__learning_rate': np.arange(0.05,1, step = 0.05),
#     'Model_fitting__n_estimators':np.arange(2,202, step = 2),
#     'Model_fitting__min_samples_split': np.arange(2,100,step =2),
#     'Model_fitting__max_depth':np.arange(1,15, step =1),
#     'Model_fitting__max_features':np.array(['sqrt','log2','auto'])
# }
# grid_cv = model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
# print("______Regression fitt_______")
# grid_cv.fit(X=regr_data, y=target)
# b_est = grid_cv.best_estimator_
# pred = b_est.predict(regr_data)
# print("MAE: {:.2f}".format(metrics.mean_absolute_error(target, pred)))
# print("RMSE: {:.2f}".format(np.sqrt(metrics.mean_squared_error(target, pred))))
# print("MAPE: {:.2f}".format(np.mean(np.abs(target - pred)/target)*100))




#
regressor = ensemble.RandomForestRegressor(n_estimators=43, random_state=1)
estimator = createEstimator(regressor)
estimator.fit(regr_data, target)
pred = estimator.predict(regr_data)
print("MAE: {:.2f}".format(metrics.mean_absolute_error(target, pred)))
print("RMSE: {:.2f}".format(np.sqrt(metrics.mean_squared_error(target, pred))))
print("MAPE: {:.2f}".format(np.mean(np.abs(target - pred)/target)*100))
#
# #_____Some cool plots______
# #______Predicted vs Real target________
plt.figure(figsize=(20,7))
plt.subplot(2,1,1)
plt.plot([min(pred),max(pred)], [min(target),max(target)], color = 'r')
plt.scatter(pred, target)
plt.ylabel("Известные значения", fontsize = 12)
plt.xlabel("Предсказанные значения", fontsize = 12)
plt.title("MAE: {:.2f}, RMSE: {:.2f}".format(metrics.mean_absolute_error(target, pred),
                                             np.sqrt(metrics.mean_squared_error(target, pred))),
          fontsize = 12, loc = 'left')
#________Densitys of real and predicted tergets_________
_,p_val = stats.wilcoxon(target,pred)
plt.subplot(2,1,2)
sns.distplot(target, hist=False, label='real')
sns.distplot(pred, hist = False, label = 'pred')
plt.title("P-value: {:.2f}".format(p_val), fontsize = 12, loc='left')
plt.legend()
plt.show()
#
#____Residual analisys____
res = target - pred
plt.figure()
plt.subplot(2,1,1)
plt.scatter(x=np.arange(1,len(res)+1,step=1),y=res)
plt.subplot(2,1,2)
sns.distplot(res)
plt.show()
#_____One more regression task______
print(stats.pearsonr(target,pred))
lm = linear_model.LinearRegression()
lm.fit(X = pred.reshape(-1,1), y = target)
final_pred = lm.predict(pred.reshape(-1,1))
print(lm.coef_)
print(lm.intercept_)
print()
est_b = sum((pred - np.mean(pred))*(final_pred - np.mean(final_pred)))/sum((pred-np.mean(pred))*(pred-np.mean(pred)))
print(est_b)
print(np.mean(target)-est_b*np.mean(pred))
#______Final Plot for ML task_____
print(target.iloc[pred == min(pred)])
target_for_plots = [target[pred == min(pred)].values,target[pred == max(pred)].values]
print(target_for_plots)
plt.figure()
plt.plot([min(pred),max(pred)],target_for_plots, color = 'r')
plt.plot([min(pred),max(pred)],[min(pred)*lm.coef_+lm.intercept_, max(pred)*lm.coef_+lm.intercept_], color = 'g')
plt.scatter(pred, final_pred)
plt.title("MAE: {:.2f}, RMSE: {:.2f}".format(metrics.mean_absolute_error(target, final_pred),
                                            np.sqrt(metrics.mean_squared_error(target, final_pred))),
          fontsize = 12, loc = 'left')
plt.xlabel("Результаты регрессионного леса")
plt.ylabel("Результаты линейной регрессии")
plt.show()
#
non_log_target = np.exp(target)
non_log_pred = np.exp(final_pred)
plt.figure()
plt.scatter(non_log_pred, non_log_target)
plt.xlabel("Предсказанные значения")
plt.ylabel("Реальные значниея")
plt.title("MAE: {:.2f}, RMSE: {:.2f}".format(metrics.mean_absolute_error(non_log_target, non_log_pred),
                                            np.sqrt(metrics.mean_squared_error(non_log_target, non_log_pred))))
plt.show()
