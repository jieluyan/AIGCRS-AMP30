from ifeature.codes import readFasta
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pycaret.utils.generic import check_metric
from ifeature.codes import *
from ifeature.PseKRAAC import *

# used_fts = ["type3Braac9", "type7raac19", "type8raac16",  "type1raac18", "type10raac18",
#                    "CKSAAP", "DDE", "type11raac10", "type12raac16", "type8raac13", "KSCTriad"]

used_fts = ["type3Braac9", "type7raac19", "CKSAAP", "DDE", "KSCTriad"]
# used_fts = ["type3Braac9", "type7raac19", "type8raac16", "CKSAAP", "KSCTriad"]

p_cai_amp = "/home/yanjielu/AMP-Gene/AMP_data/amp_cai_5_30.fasta"
p_cai_nonamp = "/home/yanjielu/AMP-Gene/AMP_data/amp_cai_random_5_30.fasta"

def SpecificityTNR(ori, pred):
    rs = confusion_matrix(ori, pred)
    [tn, fp], [fn, tp] = rs[0], rs[1]
    # tn / (tn + fp)
    return tn / (tn + fp)

subtype = {'g-gap': 0, 'lambda-correlation': 4}
def genePsekraac(path, ft_name="type1", raactype=2, subtype={'g-gap': 0, 'lambda-correlation': 4}, ktuple=2, gap_lambda=1, class_val=1):
    """
    run codes as follows if you want try all kraac features of ifeature.PseKRAAC
    from codes.ifeature.PseKRAAC import *
    from codes.ifeature.codes import readFasta
    for ft_name in kraac:
        AAGroup = eval("%s.AAGroup" % ft_name)
        for raactype in AAGroup:
    :param path: given fasta file location
    :param ft_name: one element of the following list:
        ["type1","type2","type3A","type3B","type4","type5","type6A",
         "type6B","type6C","type7","type8","type9","type10","type11",
         "type12","type13","type14","type15","type16"]
    :param raactype: key of AAGroup # of ifeature.PseKRAAC.typeX
    :param subtype: a dictionary, for example: {'g-gap': 0, 'lambda-correlation': 4}
        Note: g-gap + lambda-correlation < 5 (smallest length of sequences of given fasta file)
    :param ktuple: int 1, 2, or 3 # of ifeature.PseKRAAC.typeX
    :param gap_lambda: int 1 # of ifeature.PseKRAAC.typeX
        gap value for the ‘g-gap’ model  or lambda value for the ‘lambda-correlation’ model, 10 values are available (i.e. 0, 1, 2, ..., 9)
    :return: dataframe of feature and the last column is the label of class with name "default"
    """
    a = path.find(".csv") + path.find(".fastas")
    if a < 0:
        fastas = readFasta.readFasta(path)
    else:
        temp = pd.read_csv(path)
        if "ID" not in temp.columns:
            temp["ID"] = temp.index + 1
        if "SEQ" in temp.columns:
            temp["SEQUENCE"] = temp["SEQ"]
        fastas = list(temp.to_numpy())
    eval_func = "%s.type1(fastas, subtype, raactype, ktuple, gap_lambda)" % (ft_name)
    print(eval_func)
    encdn = eval(eval_func)
    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    print("feature number of PseKRAAC.%s(g-gap=%d, lambda-correlation=%d, raac_type=%d, ktuple=%d, gap_lambda=%d): %d" %
          (ft_name, subtype['g-gap'], subtype['lambda-correlation'], raactype, ktuple, gap_lambda, len(df.columns)))
    ft_whole_name = "%sraac%s" % (ft_name, raactype)
    print("Gene %s from path :\n\t%s" % (ft_whole_name, path))
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

def GeneIfeature(path, ft_name="AAC", nlag=4, lambdaValue=4, class_val=1):
    """
    run codes as follows if you want try all features of ifeature.codes:
    from codes.ifeature.codes import *
    for ft_name in ft_whole_type:
        if ft_name in rm_type:
            continue
    :param path: given fasta file location
    :param ft_name: each element of a list ft_whole_type:
            ft_whole_type = ["AAC","EAAC","CKSAAP","DPC","DDE","TPC","BINARY","GAAC","EGAAC",
                "CKSAAGP","GDPC","GTPC","AAINDEX","ZSCALE","BLOSUM62","NMBroto",
                "Moran","Geary","CTDC","CTDT","CTDD","CTriad","KSCTriad",
                "SOCNumber","QSOrder","PAAC","APAAC","KNNprotein","KNNpeptide",
                "PSSM","SSEC","SSEB","Disorder","DisorderC","DisorderB","ASA","TA"]
            except the list rm_type as follows:
            rm_type = ["EAAC", "BINARY", "EGAAC", "AAINDEX", "ZSCALE", "BLOSUM62", "PSSM", "ASA", "TA", "Disorder", "DisorderB",
                   "DisorderC", "KNNprotein", "KNNpeptide", "SSEC", "SSEB"]
            because of errors when run ifeature.codes.ft_name:
                Error: for "EAAC"/"BINARY"/"EGAAC"/"AAINDEX"/"ZSCALE"/"BLOSUM62"/"PSSM"/"ASA"/"TA" encoding, the input fasta sequences should be with equal length.
                "KNNprotein"/"KNNpeptide": should have the train fasta file and a label file, do it later
                "SSEC"/"SSEB": secondary structure
                "Disorder"/"DisorderB"/"DisorderC" : Protein disorder information was first predicted by the VSL2 software
                "Disorder"/"DisorderB": encoding, the input fasta sequences should be with equal length.
    :return: dataframe of feature and the last column is the label of class with name "default"
    """
    a = path.find(".csv") + path.find(".fastas")
    if a < 0:
        fastas = readFasta.readFasta(path)
    else:
        temp = pd.read_csv(path)
        if "ID" not in temp.columns:
            temp["ID"] = temp.index + 1
        if "SEQ" in temp.columns:
            temp["SEQUENCE"] = temp["SEQ"]
        temp = temp[["ID", "SEQUENCE"]]
        fastas = list(temp.to_numpy())
    # fastas = readFasta.readFasta(path)
    eval_func = "%s.%s(fastas, order=None, nlag=%d, lambdaValue=%d, gap=1)" % (ft_name, ft_name, nlag, lambdaValue)
    print(eval_func)
    encdn = eval(eval_func)
    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    print("Gene %s from path :\n\t%s" % (ft_name, path))
    row, col = df.shape
    print("\t\t its feature names: %s" % (df.columns))
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

def calMetrics(pred_rs):
    ori = pred_rs['default'].copy()
    pred = pred_rs['Label'].copy()
    Accuracy = check_metric(ori, pred, 'Accuracy')
    if 'Score_1' in pred_rs:
        sco = pred_rs['Score_1'].copy()
        AUC = check_metric(ori, sco, 'AUC')
    else:
        AUC = 0
    Recall = check_metric(ori, pred, 'Recall')
    Precision = check_metric(ori, pred, 'Precision')
    F1 = check_metric(ori, pred, 'F1')
    Kappa = check_metric(ori, pred, 'Kappa')
    rs = confusion_matrix(ori, pred)
    [tn, fp], [fn, tp] = rs[0], rs[1]
    if tp*tn - fp*fn == 0:
        MCC = 0
    else:
        MCC = check_metric(ori, pred, 'MCC')
    # MAE, MSE, RMSE, R2, RMSLE, MAPE.
    TNR = tn / (tn + fp)
    metrics_dict = {"Accuracy":Accuracy, "AUC":AUC, "Sp/TNR":TNR, "Recall/Sn/TPR":Recall, "Prec.":Precision, "F1":F1, "Kappa":Kappa, "MCC":MCC}
    metrics = pd.DataFrame([metrics_dict])
    return metrics

class MLfeaturesExtraction():
    def __init__(self, best_ft_list=used_fts, po_path=p_cai_amp, ne_path=p_cai_nonamp, init=True):
        self.best_ft_list = best_ft_list
        self.po_path, self.ne_path= po_path, ne_path
        self.ift_types = ["AAC", "CKSAAP", "DPC", "DDE", "TPC", "GAAC",
                         "CKSAAGP", "GDPC", "GTPC", "NMBroto",
                         "Moran", "Geary", "CTDC", "CTDT", "CTDD", "CTriad", "KSCTriad",
                         "SOCNumber", "QSOrder", "PAAC", "APAAC"]
        self.kraacs = ["type1", "type2", "type3A", "type3B", "type4", "type5", "type6A",
                 "type6B", "type6C", "type7", "type8", "type9", "type10", "type11",
                 "type12", "type13", "type14", "type15", "type16"]
        self.initAllFtNames()
        self.subtype = {'g-gap': 0, 'lambda-correlation': 4}
        self.ktuple = 2
        self.gap_lambda = 1
        self.nlag=4
        self.lambdaValue=4
        self.useAllFts = False
        if init:
            self.initDatasets()

    def initAllFtNames(self):
        raac_list = []
        raac_name = []
        ft_names = []
        for ft_type in self.kraacs:
            AAGroup = eval("%s.AAGroup" % ft_type)
            for raactype in AAGroup:
                if raactype < 20:
                    raac_name.append("%sraac%d" % (ft_type, raactype))
                    if raactype > 5:
                        raac_list.append([ft_type, raactype])
        self.all_kraac_list = raac_list
        self.all_kraac_typeNames = raac_name
        ft_names.extend(raac_name)
        ft_names.extend(self.ift_types)
        self.all_ft_typeNames = ft_names
        return

    def importDataByPsekraac(self, po_path, ne_path, ft_name="type1"):
        po_df = genePsekraac(po_path, ft_name, self.raactype, self.subtype, self.ktuple, self.gap_lambda, class_val=1)
        ne_df = genePsekraac(ne_path, ft_name, self.raactype, self.subtype, self.ktuple, self.gap_lambda, class_val=0)
        df = pd.concat((po_df,ne_df))
        return df

    def importDataByiFtCodes(self, po_path, ne_path, ft_name="AAC"):
        po_df = GeneIfeature(po_path, ft_name, self.nlag, self.lambdaValue, class_val=1)
        ne_df = GeneIfeature(ne_path, ft_name, self.nlag, self.lambdaValue, class_val=0)
        df = pd.concat((po_df,ne_df))
        return df

    def import1FtData(self, ft_whole_name="type1raac10"):
        # po_path, ne_path = self.po_path, self.ne_path
        if "type" in ft_whole_name:
            self.raactype = int(ft_whole_name.split("raac")[-1])
            if "Ktuple" in ft_whole_name:
                ft_name = ft_whole_name.split("Ktuple")[0]
            else:
                ft_name = ft_whole_name.split("raac")[0]
            train_df = self.importDataByPsekraac(self.po_path, self.ne_path, ft_name)
        else:
            ft_name = ft_whole_name
            train_df = self.importDataByiFtCodes(self.po_path, self.ne_path, ft_name)
        return train_df

    def ftset2ftImgAndLabel(self, df):
        label = df["default"].to_list()
        new_df = df.drop(["default"], 1)
        dict = {"img": new_df.copy(),
                "cls": label}
        return dict

    def standardInputOutput(self, sets):
        c = -1
        for ft_type in sets:
            c += 1
            if c == 0:
                ft = sets[ft_type]['img']
            else:
                temp = sets[ft_type]['img']
                ft = pd.concat([ft, temp], axis=1)
        row, col = ft.shape
        col_names = ["ft%d" % i for i in range(col)]
        ft.columns = col_names
        ft['default'] = sets[ft_type]['cls']
        return ft

    def geneAllRaacFeatureSets(self):
        train_sets = {}
        ft_list = self.best_ft_list
        in_shapes = []
        for i in range(len(ft_list)):
            # for ft_whole_name in self.best_ft_list:
            # print("Processing file:\n\t": % novel_ne_path)
            ft_whole_name = ft_list[i]
            print("Processing feature: %s" % ft_whole_name)
            train_df = self.import1FtData(ft_whole_name)
            train_dict = self.ftset2ftImgAndLabel(train_df)
            train_sets[ft_whole_name] = train_dict
        self.train_po_num, self.train_ne_num = sum(train_df["default"]), len(train_df) - sum(train_df["default"])
        print("Processing Cai's AMPs and non-AMPs (random generate):")
        print("\tTrain positive No.: ", self.train_po_num)
        print("\tTrain negative No.: ", self.train_ne_num)
        return train_sets

    def initDatasets(self):
        self.train_sets = self.geneAllRaacFeatureSets()
        self.train = self.standardInputOutput(self.train_sets)
        self.x_shape = self.train.shape
        return

def import1FtData(path, ft_whole_name="type1raac10", subtype={'g-gap': 0, 'lambda-correlation': 4},
                  ktuple=2, gap_lambda=1, nlag=4, lambdaValue=4, class_val=None):
    if "type" in ft_whole_name:
        raactype = int(ft_whole_name.split("raac")[-1])
        if "Ktuple" in ft_whole_name:
            ft_name = ft_whole_name.split("Ktuple")[0]
        else:
            ft_name = ft_whole_name.split("raac")[0]
        df = genePsekraac(path, ft_name, raactype, subtype, ktuple, gap_lambda, class_val=class_val)
    else:
        ft_name = ft_whole_name
        df = GeneIfeature(path, ft_name, nlag, lambdaValue, class_val=class_val)
    return df

def importFts(path, ft_list, class_val=None, scale_token=False):
    sets = []
    for ft_whole_name in ft_list:
        if ft_whole_name == "token":
            from geneTokens import getTokenFromCsv
            df = getTokenFromCsv(path)
            if scale_token:
                df = df / 12.5 - 1
        else:
            df = import1FtData(path, ft_whole_name, class_val=class_val)
        print("processing: %s" % ft_whole_name)
        row, col = df.shape
        col_names = [ft_whole_name + "%d" % i for i in range(col)]
        df.columns = col_names
        df.index = np.arange(row)
        sets.append(df)
    ft = pd.concat(sets, axis=1)
    return ft

def genFtsLabels():
    amp_file = '/home/yanjielu/Downloads/AMPData/AMP_recheck_len20-45.csv'
    nonamp_file = '/home/yanjielu/Downloads/AMPData/AMP_recheck_len20-45_random.csv'

    fts_name = "_".join(used_fts)
    data_dir = "~/AMP-Gene-iseLab/AMP_data/"

    amp_dfs = importFts(path=amp_file, ft_list=used_fts, class_val=None, scale_token=False)
    nonamp_dfs = importFts(path=nonamp_file, ft_list=used_fts, class_val=None, scale_token=False)
    amp_outfile = amp_file.replace(".csv", "_" + fts_name + ".csv")
    nonamp_outfile = nonamp_file.replace(".csv", "_" + fts_name + ".csv")
    amp_dfs.to_csv(amp_outfile, header=True, index=False)
    nonamp_dfs.to_csv(nonamp_outfile, header=True, index=False)
    dfs = pd.concat((amp_dfs, nonamp_dfs))
    cls = [1] * len(amp_dfs)
    cls.extend([0] * len(nonamp_dfs))

    dfs["default"] = cls
    # ft_path = "~/AMP-Gene-iseLab/AMP_data/feature/amp_cai_5_30_amp_random_classifier_features.csv"
    # ft_path = data_dir + "/feature/acp_carter_random_classifier_features.csv"
    ft_path = data_dir + "/feature/sa_cai_random_classifier_features.csv"
    ft_path = '/home/yanjielu/Downloads/AMPData/RFfts_AMP_recheck_len20-45_random.csv'
    # ft_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/feature/acp_carter_5_30_amp_random_classifier_features.csv"
    # print(dfs.columns)
    dfs.to_csv(ft_path, header=True, index=False)


def categorize_value(value):
    if value < 10:
        return 1
    elif value > 100:
        return -1
    else:
        return 0

if __name__ == "__main__":
    # used_fts = ["token", "AAC"]
    # used_fts = ["token", "AAC", "DDE"]
    p_amp = '/home/yanjielu/AMP-sets-20250417/amps_2025.csv'
    p_amp_rnd = '/home/yanjielu/AMP-sets-20250417/amps_2025_rnd.csv'
    amp_file = '~/AMP-Gene-iseLab/AMP_data/regression/EC.csv'
    nonamp_file = '~/AMP-Gene-iseLab/AMP_data/regression/SA_rnd.csv'

    df = pd.read_csv(amp_file)
    df['default'] = df['EC_MIC'].apply(categorize_value)

    fts_name = "_".join(used_fts)
    # used_fts = ["token"].extend(used_fts)
    data_dir = "~/AMP-Gene-iseLab/AMP_data/"
    # data_dir = "~/AMP-sets-20250417/"

    # amp_file = data_dir + "amp_cai_5_30.csv"
    # nonamp_file = data_dir + "amp_cai_random_5_30.csv"
    # amp_file = data_dir + "amps_2025.csv"
    # nonamp_file = data_dir + "amps_2025_rnd.csv"
    # amp_file = data_dir + "acp_carter_5_30.csv"
    # nonamp_file = data_dir + "acp_carter_random_5_30.csv"
    amp_dfs = importFts(path=amp_file, ft_list=used_fts, class_val=None, scale_token=False)
    # nonamp_dfs = importFts(path=nonamp_file, ft_list=used_fts, class_val=None, scale_token=False)
    amp_outfile = amp_file.replace(".csv", "_" + fts_name + ".csv")
    # nonamp_outfile = nonamp_file.replace(".csv", "_" + fts_name + ".csv")
    amp_dfs.to_csv(amp_outfile, header=True, index=False)
    # nonamp_dfs.to_csv(nonamp_outfile, header=True, index=False)
    # dfs = pd.concat((amp_dfs, nonamp_dfs))
    # cls = [1] * len(amp_dfs)
    # cls.extend([0] * len(nonamp_dfs))
    amp_dfs["default"] = df['default']
    ft_path = data_dir + "/feature/ec_cai_features_3labels.csv"
    amp_dfs.to_csv(ft_path, header=True, index=False)
