import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from geneTokens import getTokenFromCsv
from AMPclassifier import importFts, used_fts
p_cai_amp = "/home/yanjielu/AMP-Gene/AMP_data/amp_cai_5_30.csv"
p_cai_nonamp = "/home/yanjielu/AMP-Gene/AMP_data/amp_cai_random_5_30.csv"
p_peptide = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/PeptideAtlas_5_30_del_ampCai_random.csv"
p_gene = "/home/yanjielu/AMP-Gene-iseLab/new_result/new_amp_cai_generated_seqs_5_30_320epoch_200tiemstep_MSE.csv"

# change used_fts to only amino acid composition
# used_fts = ["AAC"]
no = 200
# no = 27180
# no = 9700
temp_p = "./temp.csv"
def rndSelNelements(no, csv_path):
    df = pd.read_csv(csv_path)
    new_df = df.sample(frac=1).copy()[0:no]
    new_df.to_csv(temp_p, header=True, index=False)
    return

rndSelNelements(no, p_cai_amp)
cai_amp_token = getTokenFromCsv(p_cai_amp)
amp_dfs = importFts(path=temp_p, ft_list=used_fts, class_val=None, scale_token=False)

rndSelNelements(no, p_cai_nonamp)
cai_nonamp_token = getTokenFromCsv(temp_p)
nonamp_dfs = importFts(path=temp_p, ft_list=used_fts, class_val=None, scale_token=False)

rndSelNelements(no, p_gene)
gene_token = getTokenFromCsv(temp_p)
gene_dfs = importFts(path=temp_p, ft_list=used_fts, class_val=None, scale_token=False)

rndSelNelements(no, p_peptide)
pep_token = getTokenFromCsv(temp_p)
pep_dfs = importFts(path=temp_p, ft_list=used_fts, class_val=None, scale_token=False)



tokens = []
tokens.append(cai_amp_token.sample(frac=1).copy()[0:no])
tokens.append(cai_nonamp_token.sample(frac=1).copy()[0:no])
tokens.append(pep_token.sample(frac=1).copy()[0:no])
tokens.append(gene_token.sample(frac=1).copy()[0:no])


fts = []
fts.append(amp_dfs.sample(frac=1).copy()[0:no])
fts.append(nonamp_dfs.sample(frac=1).copy()[0:no])
fts.append(pep_dfs.sample(frac=1).copy()[0:no])
fts.append(gene_dfs.sample(frac=1).copy()[0:no])


all = pd.concat(fts)
all_label = [0]*no
all_label.extend([1]*no)
all_label.extend([2]*no)
all_label.extend([3]*no)

target_names = {
    0:'Real',
    1:'Random',
    2:'Peptide',
    3:'Generate'
}
clrs = ['cadetblue', 'coral', 'royalblue', 'lightseagreen']
# clrs = ['cadetblue', 'coral', 'royalblue', 'indianred']
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# all_scaled = StandardScaler().fit_transform(all)
all_scaled = all.copy()/25
pca = PCA(n_components=2)
pca_features = pca.fit_transform(all_scaled)
# Create dataframe
pca_df = pd.DataFrame(
    data=pca_features,
    columns=['PC1', 'PC2'])
pca_df['target'] = all_label
pca_df['target'] = pca_df['target'].map(target_names)

# plot all labels
sns.lmplot(
    x='PC1',
    y='PC2',
    # z='PC3',
    data=pca_df,
    hue='target',
    # col='target',
    palette=clrs,
    fit_reg=False,
    legend=True
)
# plt.title('Real, Random, Generated AMPs, and AtlaPeps PCA Scatter Plot')
plt.show()

# pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/pca_plot_cai_amp_random_generate_AMPs_token_%d.png" % no
pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/4pca_plot_cai_amp_random_generate_AMPs_AtlaPeps_token_%d.png" % no
# pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/pca_plot_cai_amp_random_generate_AMPs_usedfts_%d.png" % no
pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/4pca_plot_cai_amp_random_generate_AMPs_AtlaPeps_usedfts_%d.png" % no
# pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/pca_plot_cai_amp_random_generate_AMPs_AAC_%d.png" % no
# pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/4pca_plot_cai_amp_random_generate_AMPs_AtlaPeps_AAC_%d.png" % no
plt.savefig(pic_path, dpi=300, format="png")

# put all labels to subplots
sns.lmplot(
    x='PC1',
    y='PC2',
    # z='PC3',
    data=pca_df,
    col='target',
    palette=clrs,
    fit_reg=False,
    legend=True
)
# plt.title('Real, Random, Generated AMPs, and AtlaPeps PCA Scatter Plot')
plt.show()

# pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/pca_plot_cai_amp_random_generate_AMPs_3subfigs_token_%d.png" % no
pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/4pca_plot_cai_amp_random_generate_AMPs_AtlaPeps_4subfigs_token_%d.png" % no
# pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/pca_plot_cai_amp_random_generate_AMPs_3subfigs_usedfts_%d.png" % no
pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/4pca_plot_cai_amp_random_generate_AMPs_AtlaPeps_4subfigs_usedfts_%d.png" % no
# pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/pca_plot_cai_amp_random_generate_AMPs_3subfigs_AAC_%d.png" % no
# pic_path = "/home/yanjielu/AMP-Gene-iseLab/AMP_data/pics/4pca_plot_cai_amp_random_generate_AMPs_AtlaPeps_4subfigs_AAC_%d.png" % no
plt.savefig(pic_path, dpi=300, format="png")

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# plt.title('3D Scatter Plot with Seaborn')
# plt.show()