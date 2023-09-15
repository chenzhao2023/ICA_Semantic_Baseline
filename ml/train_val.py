import warnings
import os
import argparse
from numpy.lib.function_base import average
import pandas as pd
warnings.filterwarnings('ignore')
import numpy as np


from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def report_performance(clf, train_x, train_y, test_x, test_y, thresh=0.5, clf_name="CLF"):
    print("[x] performance for {} classifier".format(clf_name))
    y_train_preds = clf.predict_proba(train_x)[:, 1]
    y_test_preds = clf.predict_proba(test_x)[:, 1]
    print('Training:')
    train_accuracy, train_recall, train_precision = print_report(train_y, y_train_preds, thresh)
    print('Test:')
    test_accuracy, test_recall, test_precision = print_report(test_y, y_test_preds, thresh)
    return {"train": {"acc": train_accuracy, "recall": train_recall, "precision": train_precision},
            "test": {"acc": test_accuracy, "recall": test_recall, "precision": test_precision}}


def print_report(y_actual, y_pred, thresh):
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh), average='weighted')
    precision = precision_score(y_actual, (y_pred > thresh), average='weighted')
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print(' ')
    return accuracy, recall, precision

def train_svm(train_x, train_y, test_x, test_y, n_split=5):
    svc = SVC(probability=True)
    parameters = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                  'gamma': [0.001, 0.0001],
                  'kernel': ['linear', 'rbf', 'sigmoid']}
    scoring = make_scorer(accuracy_score)
    grid_clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=n_split, n_jobs=-1, verbose=0, scoring=scoring)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    report_performance(grid_clf.best_estimator_, train_x, train_y, test_x, test_y, clf_name="SVM")
    return grid_clf.best_estimator_, grid_clf.best_params_


def train_sgd(train_x, train_y, test_x, test_y, n_split=5):
    sgdc = SGDClassifier(loss='log', alpha=0.1, random_state=42)
    penalty = ['none', 'l2', 'l1']
    max_iter = range(100, 500, 100)
    alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    random_grid_sgdc = {'penalty': penalty, 'max_iter': max_iter, 'alpha': alpha}
    scoring = make_scorer(accuracy_score)
    grid_clf = GridSearchCV(estimator=sgdc, param_grid=random_grid_sgdc, cv=n_split, scoring=scoring, verbose=0)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    report_performance(grid_clf.best_estimator_, train_x, train_y, test_x, test_y, clf_name="SGD")
    return grid_clf.best_estimator_, grid_clf.best_params_


def train_gradient_boosting(train_x, train_y, test_x, test_y, n_split=5):
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=42, )

    n_estimators = range(5, 100, 5)
    max_depth = range(1, 10, 1)
    learning_rate = [0.001, 0.01, 0.1]
    random_grid_gbc = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}

    scoring = make_scorer(accuracy_score)
    grid_clf = GridSearchCV(estimator=gbc, param_grid=random_grid_gbc, cv=n_split, scoring=scoring, verbose=0)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    report_performance(grid_clf.best_estimator_, train_x, train_y, test_x, test_y, clf_name="Gradient Boosting")
    return grid_clf.best_estimator_, grid_clf.best_params_


def train_knn(train_x, train_y, test_x, test_y, n_split=5):
    knn = KNeighborsClassifier(n_neighbors=10)
    n_neighbors = range(3, 30)
    weights = ['uniform', 'distance']
    random_grid_knn = {'n_neighbors': n_neighbors, 'weights': weights}
    scoring = make_scorer(accuracy_score)
    grid_clf = GridSearchCV(estimator=knn, param_grid=random_grid_knn, cv=n_split, scoring=scoring, verbose=0)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    report_performance(grid_clf.best_estimator_, train_x, train_y, test_x, test_y, clf_name="KNN")
    return grid_clf.best_estimator_, grid_clf.best_params_


def train_lr(train_x, train_y, test_x, test_y, n_split=5):
    lr = LogisticRegression(random_state=42)
    parameters = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'penalty': ["l2"]}
    #auc_scoring = make_scorer(roc_auc_score, multi_class='ovo')
    scoring = make_scorer(accuracy_score)
    grid_clf = GridSearchCV(estimator=lr, param_grid=parameters, cv=n_split, scoring=scoring, verbose=0)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    report_performance(grid_clf.best_estimator_, train_x, train_y, test_x, test_y, clf_name="LR")
    return grid_clf.best_estimator_, grid_clf.best_params_


def train_rf(train_x, train_y, test_x, test_y, n_split=5):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(max_depth=6, random_state=42)
    rf.fit(train_x, train_y)

    n_estimators = range(5, 100, 5)
    max_features = ['auto', 'sqrt']
    max_depth = range(1, 10, 1)
    min_samples_split = range(2, 10, 2)
    criterion = ['gini', 'entropy']
    parameters = {'n_estimators': n_estimators, 'max_features': max_features,
                  'max_depth': max_depth, 'min_samples_split': min_samples_split, 'criterion': criterion}
    scoring = make_scorer(accuracy_score)
    grid_clf = GridSearchCV(estimator=rf, param_grid=parameters, cv=n_split, scoring=scoring, verbose=0, n_jobs=-1)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    report_performance(grid_clf.best_estimator_, train_x, train_y, test_x, test_y, clf_name="RF")
    return grid_clf.best_estimator_, grid_clf.best_params_


def train_mlp(train_x, train_y, test_x, test_y, n_split=5):
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier()

    activation = ['identity', 'logistic', 'tanh', 'relu']
    hidden_layer_sizes = [(10, 5), (20, 10), (5, 10), (10,), (5,), (15,)]
    alpha = [0.0001, 0.001, 0.01, 0.1, 1]
    parameters = {'activation': activation, 'hidden_layer_sizes': hidden_layer_sizes,'alpha': alpha}
    auc_scoring = make_scorer(roc_auc_score)
    grid_clf = GridSearchCV(estimator=mlp, param_grid=parameters, cv=n_split, scoring=auc_scoring, verbose=0, n_jobs=-1)
    grid_clf.fit(train_x, train_y)

    print(grid_clf.best_estimator_)
    print(grid_clf.best_params_)
    report_performance(grid_clf.best_estimator_, train_x, train_y, test_x, test_y, clf_name="MLP")
    return grid_clf.best_estimator_, grid_clf.best_params_


def get_split_deterministic(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """
    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/home/zhaochen/Desktop/vessel_semantic_segmentation")
    parser.add_argument('--data_split', type=str, default="data_nj_new/data_split")
    parser.add_argument('--data_root', type=str, default="data_nj_new/processed")
    parser.add_argument('--save_dir', type=str, default="data_nj_new/semantic_data")
    parser.add_argument('--output_csv', type=str, default="feature.csv")

    args = parser.parse_args()

    feature_df = pd.read_csv(os.path.join(args.base_path, args.save_dir, args.output_csv))

    classes = sorted(feature_df['class'].unique())
    # add new colume label as the integer labels for each segment
    label_maker = LabelEncoder()
    feature_df['label'] = label_maker.fit_transform(feature_df['class'])

    # add lao rao
    values = []
    for row in feature_df['patient_name']:
        if row.rfind("LAO") != -1:
            values.append(0)
        else:
            values.append(1)

    # add one-hot encoder
    one_hot_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    one_hot_enc.fit(feature_df[['label']])
    #feature_df['one_hot'] = one_hot_enc.fit_transform(feature_df[['label']])

    train_df = feature_df[feature_df['patient_name'].isin(train_patients)]
    test_df = feature_df[feature_df['patient_name'].isin(test_patients)]

    print(f"split entire datasheet into a training set {train_df.shape} and a testing set {test_df.shape}")

    train_x = train_df.filter(regex=("feature*"))
    test_x = test_df.filter(regex=("feature*"))
    #train_y = one_hot_enc.transform(train_df[['label']])
    #test_y = one_hot_enc.transform(test_df[['label']])
    train_y = np.squeeze(np.array(train_df[['label']]))
    test_y = np.squeeze(np.array(test_df[['label']]))
    train_lr(train_x, train_y, test_x, test_y, n_split=3)
    train_knn(train_x, train_y, test_x, test_y, n_split=3)
    train_rf(train_x, train_y, test_x, test_y, n_split=3)
    train_sgd(train_x, train_y, test_x, test_y, n_split=3)
    train_svm(train_x, train_y, test_x, test_y, n_split=3)
    train_gradient_boosting(train_x, train_y, test_x, test_y, n_split=3)
    #train_mlp(tr_x[i], tr_y[i], te_x[i], te_y[i], cv)