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

# ft_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/feature/amp_cai_5_30_amp_random_classifier_features.csv"
# ft_path = "~/AMP-sets-20250417/feature/amp_cai_5_30_amp_random_classifier_features.csv"
ft_path = "~/AMP-Gene-iseLab/AMP_data/feature/acp_carter_5_30_amp_random_classifier_features.csv"
ft_path = "~/AMP-Gene-iseLab/AMP_data/feature/ec_cai_random_classifier_features.csv"
ft_path = "~/AMP-Gene-iseLab/AMP_data/feature/sa_cai_random_classifier_features.csv"
ft_path = "~/AMP-Gene-iseLab/AMP_data/feature/ec_cai_features_3labels.csv"
ft_path = "~/AMP-Gene-iseLab/AMP_data/feature/sa_cai_features_3labels.csv"
ft_path = "~/Downloads/AMPData/RFfts_AMP_recheck_len20-45_random.csv"

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

# rf_amp_mdl="/home/yanjielu/AMP-Gene-iseLab/model/new_amp_random_forest_ntree2000_log2_reverse.joblib"
# rf_amp_mdl="/home/yanjielu/AMP-sets-20250417/model/amp2025_random_forest_ntree2000_log2_reverse.joblib"
# rf_acp_mdl="/home/yanjielu/AMP-Gene-iseLab/model/new_acp_random_forest_ntree1800_log2_reverse.joblib"
# rf_ec_mdl="/home/yanjielu/AMP-Gene-iseLab/model/new_ec_random_forest_ntree1800_log2_reverse.joblib"
# rf_sa_mdl="/home/yanjielu/AMP-Gene-iseLab/model/new_sa_random_forest_ntree1800_log2_reverse.joblib"
rf_ec_mdl="/home/yanjielu/AMP-Gene-iseLab/model/ec_3labels_forest_ntree2000_log2_rgr.joblib"
rf_sa_mdl="/home/yanjielu/AMP-Gene-iseLab/model/sa_3labels_forest_ntree1800_log2_rgr.joblib"
rf_amp_mdl_len20_45 = "/home/yanjielu/Downloads/AMPData/rf_amp_mdl_len20_45_ntree2000_log2.joblib"
def RFDevelopAndTest(X, y, test_path, ntree=2000, max_features="log2",
                     rf_mdl=rf_amp_mdl_len20_45):
# def RFDevelopAndTest(X, y, test_path, ntree=1800, max_features="log2",
#                          rf_mdl=rf_acp_mdl):
    # from AMPclassifier import genePsekraac, used_fts
    # po_df = genePsekraac(po_path, ft_name, self.raactype, self.subtype, self.ktuple, self.gap_lambda, class_val=None)
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
    return rf, y_pre, prob[:,1]


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
    # RFtuneNtreesFts()

    folder = "~/AMP-Gene-iseLab/rf_new_rs/Datasizes/Classifier_guidance/"
    folder = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/regression/"
    # folder = "/home/yanjielu/AMP-Gene-iseLab/result/ACP"
    # folder = "/home/yanjielu/AMP-Gene-iseLab/checkpoints/train/2025amps"
    # folder = "/home/yanjielu/AMP-Gene-iseLab/rf_new_rs/Datasizes/"
    # folder = "/home/yanjielu/AMP-Gene-iseLab/rf_new_rs/noises/Classifier_guidance/"

    # import pathlib
    import glob

    # glob.glob('./*.csv')
    # rf_changed_ClassifierGuidance_acp_carter - Diffusion
    # b = glob.glob(folder + '/*.rep*[0-9]*')
    b = []
    for i in range(22):
        tf = "/home/yanjielu/AMP-Gene-iseLab/rf_new_rs/Datasizes/Classifier_guidance/50MClassifierGuidance_amp_cai-Diffusion_5000epoch_700tiemstep_10.00scale_clip3000epochcls.csv.rep%d" % i
        b.append(tf)
    # "rf_reversed_amp_cai_generated_seqs_5_30_5000epoch_700tiemstep_MSE_dataSize20000_rep1"
    # rf, y_pre, prob = RFDevelopAndTest(X, y)
    # b = ["/home/yanjielu/Downloads/jianxiuAMPdata/best_solutions_sa37_uni.csv"]
    b = ["/home/yanjielu/Downloads/AMPData/generated_amp_unguided.csv"]
    amp_all = []
    # np.arange(100, 1100, 100)
    for ts in [1]:
    # for guidance_scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25,
    #                         50, 75, 100]:
    #     for rep in [1]:
        for test_path in b:
            # folder = "~/AMP-Gene-iseLab/rf_new_rs/5000epochs/"
            # # test_path = ('~/AMP-Gene-iseLab/result/gene_seqs/Classifier_guidance/'
            # #              'rf_changed_ClassifierGuidance-Diffusion_5000epoch_700tiemstep_%.2fscale_clip600cls.csv') % guidance_scale
            # test_path = "/home/yanjielu/Downloads/jianxiuAMPdata/best_solutions_p20_uni.csv"

            rf, y_pre = RFDevelopAndTest(X, y, test_path=test_path, ntree=2000, max_features="log2")
            # test_path = str(test_path)
            p_csv = test_path
            if ".fasta" in test_path:
                p_csv = test_path.replace(".fasta", ".csv")
            df = pd.read_csv(test_path)
            # df["sa_class"] = y_pre
            df["AMP_prob"] = y_pre
            # df["ACP_class"] = y_pre
            # df["ACP_prob"] = prob
            s = pd.Series(df["SEQUENCE"])
            lens = s.str.len()
            df["Length"] = lens
            p_generated_seq = p_csv + ".rfAMpPre"
            # p_generated_seq = p_csv + ".rfAcpPre"
            df.to_csv(p_generated_seq, header=True, index=False)
            df90 = df[df["sa_prob"] > 0]
            # df90 = df[df["sa_prob"] > 0]
            # df90 = df[df["ACP_prob"] > 0.9]
            amp_all.append(df90)
            l = len(df)
            acc = sum(y_pre) / l * 100
            # print("Valid sequences of %d epoch %d timestep: " % (epoch, time_steps), l)
            print("Valid sequences of %s: " % (test_path), l)
            print("Accuracy: ", acc)
    p_generated_amp90 = folder + "Length26AMPProb0EC3LblSA3Lbl.csv"
    amp_all_df = pd.concat(amp_all)
    amp_all_df.to_csv(p_generated_amp90, header=True, index=False)


    # mets = []
    # for i in [1,2,3,4,5]:
    #     metric = RFXCV(X, y, 2000, "log2")
    #     mets.append(metric)
    # mets_df = pd.concat(mets)
    # mets_df = mets_df.drop(["alg_name"], axis=1)
    # mets_dfs = pd.concat([mets_df, mets_df.mean().to_frame().T, mets_df.std().to_frame().T])
    # mets_dfs.index = ["rep1", "rep2", "rep3", "rep4", "rep5", "mean", "std"]
    # # p_metrics = "~/AMP-Gene-iseLab/AMP_data/Classifier_rs/ACP_Carter_RF2000_log2_10CV_metrics.csv"
    # p_metrics = "~/AMP-Gene-iseLab/AMP_data/Classifier_rs/AMP_Cai_RF2000_log2_10CV_metrics.csv"
    # mets_dfs.to_csv(p_metrics, header=True, index=True)

    # Valid sequences of ./test/short_predict.csv:  50295
    # Accuracy:  62.31036882393877
    # Valid sequences of ./test/sa_target_len23.csv:  300
    # Accuracy:  42.333333333333336
    # Valid sequences of ./test/ec_target_len17.csv:  300
    # Accuracy:  92.0