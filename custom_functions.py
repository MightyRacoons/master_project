# _____custom functions file_____
# _____imports_____
import matplotlib.pyplot as plt
import numpy as np

from sklearn import pipeline, preprocessing


def plot_2d_data(data, target, colors):
    plt.figure(figsize=[8, 8])
    x = data[0]
    plt.scatter(x[:, 0], x[:, 1], c=target, cmap=colors)
    plt.show()


def plot_d_surface(estimator, tr_data, tr_labels, t_data, t_labels, colors, light_colors):
    estimator.fit(tr_data, tr_labels)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    xx, yy = np.meshgrid(tr_data[:, 0], tr_data[:, 1])
    mesh_pred = np.array(estimator.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_pred, cmap=light_colors)
    plt.scatter(tr_data[:, 0], tr_data[:, 1], c=tr_labels, cmap=colors)
    plt.title('Train data, accuracy={:.2f}'.format(metrics.accuracy_score(tr_labels, estimator.predict(tr_data))))
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, mesh_pred, cmap=light_colors)
    plt.scatter(t_data[:, 0], t_data[:, 1], c=t_labels, cmap=colors)
    plt.title('Test data, accuracy={:.2f}'.format(metrics.accuracy_score(t_labels, estimator.predict(t_data))))
    plt.show()


def month_mapper(month):
    m = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12}
    s = month.strip()[:3].lower()
    return m[s]


def day_mapper(day):
    d = {
        'mon': 1,
        'tue': 2,
        'wed': 3,
        'thu': 4,
        'fri': 5,
        'sat': 6,
        'sun': 7
    }
    s = day.strip()[:3].lower()
    return d[s]


def create_estimator(ml_obj, numeric_features, cat_features, date_features):
    estimator = pipeline.Pipeline(steps=[
        ('Feature_processing', pipeline.FeatureUnion(transformer_list=[
            ('Numeric_features', pipeline.Pipeline(steps=[
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_features], validate=True)),
                ('scaling', preprocessing.StandardScaler(with_mean=0., with_std=1))
            ])),
            ('Categical_features', pipeline.Pipeline(steps=[
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, cat_features], validate=True)),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore'))
            ])),
            ('Date_features', pipeline.Pipeline(steps=[
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, date_features], validate=True)),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore'))
            ]))
        ])),
        ('Model_fitting', ml_obj)
    ])
    return estimator
#TODO:
#make custom score