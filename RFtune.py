import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, cohen_kappa_score, matthews_corrcoef
from AMPclassifier import importFts, used_fts
import joblib
import os

def are_all_elements_equal(arr):
    return all(element == arr[0] for element in arr)

def calMetrics(ori, pred, probs):
    # if probs == None:
    #     probs = pred.copy()
    #     y_pre[y_pre >= 0.5] = 1
    #     y_pre[y_pre < 0.5] = 0
    Accuracy = accuracy_score(ori, pred)
    ori_auc, probs_auc = ori.copy(), probs.copy()
    if are_all_elements_equal(ori):
        add_value = 1 - ori[0]
        ori_auc = np.append(ori, [add_value])
        # pred = np.append(pred, [add_value])
        probs_auc = np.append(probs, [add_value])
    AUC = roc_auc_score(ori_auc, probs_auc)
    Recall = recall_score(ori, pred)
    Precision = precision_score(ori, pred)
    F1 = f1_score(ori, pred)
    Kappa = cohen_kappa_score(ori, pred)
    MCC = matthews_corrcoef(ori, pred)
    # metrics_dict = {"Accuracy":Accuracy, "AUC":AUC, "Sp/TNR":TNR, "Recall/Sn/TPR":Recall, "Prec.":Precision, "F1":F1, "Kappa":Kappa, "MCC":MCC}
    metrics_dict = {"Accuracy":Accuracy, "AUC":AUC, "Recall":Recall, "Prec.":Precision, "F1":F1, "Kappa":Kappa, "MCC":MCC}
    metrics = pd.DataFrame([metrics_dict])
    return metrics

ft_path = "./AMP_data/feature/amp_cai_5_30_amp_random_classifier_features.csv"

p_cai_classifier_features = ft_path
train = pd.read_csv(p_cai_classifier_features)
# shuffle rows
train = train.sample(frac=1)
X, y = train.drop(['default'], axis=1), train['default'].tolist()

# sh.best_estimator_

def RFClassifier(X, y, train_ind, test_ind, ntree=2000, max_features="log2"):
    X_train, y_train = X.iloc[train_ind, :], np.take(y, train_ind)
    X_test, y_test = X.iloc[test_ind, :], np.take(y, test_ind)
    rf = RandomForestClassifier(n_estimators=ntree, max_features=max_features, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pre = rf.predict(X_test)
    prob = rf.predict_proba(X_test)
    return y_test, y_pre, prob[:, 1]

rf_amp_mdl="./model/new_amp_random_forest_ntree2000_log2_reverse.joblib"

def RFDevelopAndTest(test_path, ntree=2000, max_features="log2",
                     X=X, y=y, rf_mdl=rf_amp_mdl):
    fts = importFts(test_path, ft_list=used_fts, class_val=None)
    if os.path.isfile(rf_mdl):
        rf = joblib.load(rf_mdl)
    else:
        rf = RandomForestClassifier(n_estimators=ntree, max_features=max_features, random_state=0, n_jobs=-1)
        # rf = RandomForestRegressor(n_estimators=ntree, max_features=max_features, random_state=0, n_jobs=-1)
        rf.fit(X, y)
        joblib.dump(rf, rf_mdl)
    y_pre = rf.predict(fts)
    prob = rf.predict_proba(fts)
    # joblib.load(rf, "./random_forest.joblib")
    return y_pre, prob[:,1]


def RFXCV(X, y, ntree=2000, max_features="log2", cv=10):
    kf = KFold(n_splits=cv)
    test_pre, test_ori, probs = [], [], []
    for train_ind, test_ind in kf.split(X, y):
        y_test, y_pre, prob = RFClassifier(X, y, train_ind, test_ind, ntree=ntree, max_features=max_features)
        test_pre.extend(y_pre)
        test_ori.extend(y_test)
        probs.extend(prob)
    metric = calMetrics(test_ori, test_pre, probs)
    alg_name = "ntree%d_%s" % (ntree, max_features)
    metric.insert(0, "alg_name", [alg_name])
    return metric

def RFtuneNtreesFts():
    import joblib
    rf_grid = [{'n_estimators': np.arange(100, 2100, 100), 'max_features': ['sqrt', 'log2']}]
    mets = []
    # for ntree in np.arange(100, 2100, 100):
    for max_features in ['sqrt', 'log2']:
        # met = RFXCV(X, y, ntree=ntree, max_features=max_features)
        ans = joblib.Parallel(n_jobs=10)(joblib.delayed(RFXCV)(X, y, ntree=ntree, max_features=max_features) for ntree in np.arange(100, 2100, 100))
        met = pd.concat(ans)
        print(mets)
        mets.append(met)
    mets_df = pd.concat(mets)
    output_filepath = "/home/yanjielu/AMP-Gene-iseLab/result/ACP/acp-carter_5_30_amp_random_tune_rf_metrics.csv"
    mets_df.to_csv(output_filepath, header=True, index=False)
    final_df  = mets_df.sort_values(by="Accuracy", ascending=False)
    output_filepath = "/home/yanjielu/AMP-Gene-iseLab/result/ACP/acp-carter_5_30_amp_random_tune_rf_metrics_sort.csv"
    final_df.to_csv(output_filepath, header=True, index=False)

if __name__ == "__main__":
    test_files = ["./sample.csv"]
    amp_all = []
    for test_path in test_files:
        rf, y_pre = RFDevelopAndTest(X, y, test_path=test_path, ntree=2000, max_features="log2")
        p_csv = test_path
        if ".fasta" in test_path:
            p_csv = test_path.replace(".fasta", ".csv")
        df = pd.read_csv(test_path)
        df["AMP_prob"] = y_pre
        s = pd.Series(df["SEQUENCE"])
        lens = s.str.len()
        df["Length"] = lens
        p_generated_seq = p_csv + ".rfAMpPre"
        df.to_csv(p_generated_seq, header=True, index=False)
        amp_all.append(df)
    amp_prob_path = "./Tested_AMP_prob.csv"
    amp_all_df = pd.concat(amp_all)
    amp_all_df.to_csv(amp_prob_path, header=True, index=False)
