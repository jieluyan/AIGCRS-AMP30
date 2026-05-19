import numpy as np
import pandas as pd
import tensorflow as tf
import dill

latent_node = 25
aa_codes = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] # the same order with aac of propy3
aa_dict = dict(zip(aa_codes, np.arange(1,21,1)))
MAX_LEN = 30 + 1
def getToken(seq, max_len=MAX_LEN, aa_codes=aa_codes):
    l = len(seq)
    token = [24] * max_len
    # print(seq)
    for i in range(l):
        token[i] = aa_codes.index(seq[i]) + 1
        # convert aa_codes to from 1 to 20, use this 20 int numbers represent all natrual amino acids
    # end point convert to 25, pad 24 as nothing from then on untill max 50 length
    token[l] = 25
    token = np.array(token)
    token = token.reshape((1, -1))
    return token

def getTokens(seqs, max_len=MAX_LEN, min_len=5, aa_codes=aa_codes, class_val=None):
    # fastas = getTrueFastas(fastas)
    fts = []
    # names = fastas[:,0]
    for seq in seqs:
        if len(seq) > max_len or len(seq) < min_len:
            continue
        e = getToken(seq, max_len=max_len, aa_codes=aa_codes)
        fts.append(e)
    fts = np.concatenate(fts, axis=0)
    df = pd.DataFrame(fts)
    print("Gene Token:")
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(df)
    return df

def getTokenFromCsv(file_path):
    df = pd.read_csv(file_path)
    seqs = df["SEQUENCE"]
    tokens = getTokens(seqs, max_len=MAX_LEN, min_len=5, class_val=None)
    return tokens

def token2seq(token, aa_codes=aa_codes, scaled=True):
    seq = ""
    if tf.is_tensor(token):
        token = token.numpy()
    for t in token:
        if scaled:
            t = int(round((t+1)*12.5))
        else:
            t = int(t)
        if t > 20:
            break
        seq = seq + aa_codes[t-1]
    return seq



def tokens2seqs(tokens, scaled=True):
    seqs = []
    for token in tokens:
        seq = token2seq(token, scaled=scaled)
        seqs.append(seq)
    return seqs

def write_pkl(obj, file_path):
    with open(file_path, 'wb') as f:
        dill.dump(obj, f)
    return

def read_pkl(file_path):
    with open(file_path, "rb") as f:
        final_rs = dill.load(f)
    return final_rs

if __name__=="__main__" :
    # # generate tokens from AMPs of Cai
    # p_cai_amp = "/home/yanjielu/AMP-Gene/AMP_data/amp_cai_5_30.csv"
    # p_cai_amp_token = "/home/yanjielu/AMP-Gene/AMP_data/token/amp_cai_5_30.csv"
    # cai_amp_token = getTokenFromCsv(p_cai_amp)
    # cai_amp_token.to_csv(p_cai_amp_token, header=True, index=False)
    #
    # # generate tokens from random sequences i.e. non-AMPs dataset of cai_amp
    # p_cai_nonamp = "/home/yanjielu/AMP-Gene/AMP_data/amp_cai_random_5_30.csv"
    # p_cai_nonamp_token = "/home/yanjielu/AMP-Gene/AMP_data/token/amp_cai_random_5_30.csv"
    # cai_nonamp_token = getTokenFromCsv(p_cai_nonamp)
    # cai_nonamp_token.to_csv(p_cai_nonamp_token, header=True, index=False)
    #
    # # generate tokens from all peptides of peptide Atlas and del amp and non-amp data of cai
    # p_all_peptide = "/home/yanjielu/AMP-Gene/AMP_data/PeptideAtlas_5_30_del_ampCai_random.csv"
    # p_all_peptide_token = "/home/yanjielu/AMP-Gene/AMP_data/token/PeptideAtlas_5_30_del_ampCai_random_token.csv"
    # all_peptide_token = getTokenFromCsv(p_all_peptide)
    # all_peptide_token.to_csv(p_all_peptide_token, header=True, index=False)

    # generate tokens from carter's acp data only remain 5 to 30 residues
    # p_acp = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/acp_carter_5_30.csv"
    # p_acp_token = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/token/acp_carter_5_30_token.csv"
    # acp_token = getTokenFromCsv(p_acp)
    # acp_token.to_csv(p_acp_token, header=True, index=False)

    # p_acp_rnd = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/acp_carter_random_5_30.csv"
    # p_acp_rnd_token = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/token/acp_carter_random_5_30_token.csv"
    # acp_rnd_token = getTokenFromCsv(p_acp_rnd)
    # acp_rnd_token.to_csv(p_acp_rnd_token, header=True, index=False)

    p_amp = '/home/yanjielu/AMP-sets-20250417/amps_2025.csv'
    p_amp_token = '/home/yanjielu/AMP-sets-20250417/token/amps_2025_token.csv'
    p_amp_rnd = '/home/yanjielu/AMP-sets-20250417/amps_2025_rnd.csv'
    p_amp_rnd_token = '/home/yanjielu/AMP-sets-20250417/token/amps_2025_rnd_token.csv'
    amp_rnd_token = getTokenFromCsv(p_amp_rnd)
    amp_rnd_token.to_csv(p_amp_rnd_token, header=True, index=False)
    amp_token = getTokenFromCsv(p_amp)
    amp_token.to_csv(p_amp_token, header=True, index=False)
