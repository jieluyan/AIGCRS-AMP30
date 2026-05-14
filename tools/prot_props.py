from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import peptides


def cal_probs(csv_path, out_path, col_names, seq_type_name):
    seq_data = pd.read_csv(csv_path)
    seq_data["molecular_weight"] = [ProteinAnalysis(i).molecular_weight() for i in seq_data["SEQUENCE"]]
    seq_data["isoelectric_point"] = [ProteinAnalysis(i).isoelectric_point() for i in seq_data["SEQUENCE"]]
    # gravy, grand average of hydropathy
    seq_data["gravy"] = [ProteinAnalysis(i).gravy() for i in seq_data["SEQUENCE"]]
    seq_data["charge"] = [ProteinAnalysis(i).charge_at_pH(7) for i in seq_data["SEQUENCE"]]
    seq_data["hydrophobic_moment"]= [peptides.Peptide(i).hydrophobic_moment()for i in seq_data["SEQUENCE"]]
    seq_type = [seq_type_name] *len(seq_data["SEQUENCE"])
    seq_data["seq_type"] = seq_type
    seq_data.to_csv(out_path, index=False)
    seq_box_data = seq_data[col_names]
    return seq_box_data

col_names = ["SEQUENCE", "molecular_weight", "isoelectric_point", "gravy", "charge","hydrophobic_moment", "seq_type"]
random_len11_ini = cal_probs("/home/yanjielu/AMP-Gene/Diffusion.rfPre",
                             "/home/yanjielu/AMP-Gene/Diffusion-prob.csv",
                             col_names, "diffusion")
len23_ini = cal_probs("/home/yanjielu/AMP-Gene/Diffusion-classifier-guidance.rfPre",
                      "/home/yanjielu/AMP-Gene/Diffusion-classifier-guidance-prob.csv",
                      col_names, "diffusion-classifier")
ec_sa_common = cal_probs("/home/yanjielu/AMP-Gene/Diffusion-classifier-free.rfPre",
                         "/home/yanjielu/AMP-Gene/Diffusion-classifier-free-prob.csv",
                         col_names, "diffusion-classifier-free")
amp_common = cal_probs("/home/yanjielu/AMP-Gene/AMP_data/amp_cai_5_30.csv",
                         "/home/yanjielu/AMP-Gene/AMP_data/amp_cai_5_30-prob.csv",
                         col_names, "True AMP")
rnd_common = cal_probs("/home/yanjielu/AMP-Gene/AMP_data/amp_cai_random_5_30.csv",
                         "/home/yanjielu/AMP-Gene/AMP_data/amp_cai_random_5_30-prob.csv",
                         col_names, "Random Non-AMP")
paths = ["~/Downloads/Diffusion_predict.csv",
 "~/Downloads/Diffusion-classifier-guidance_predict.csv",
 "~/Downloads/Diffusion-classifier-free_predict.csv"]
box_data = pd.concat([random_len11_ini, len23_ini, ec_sa_common, amp_common, rnd_common], axis=0)

colnames = ["SEQUENCE", "ec_predicted_MIC_μM", "sa_predicted_MIC_μM", "seq_type"]
seq_types = ["diffusion", "diffusion-classifier", "diffusion-classifier-free"]
mics = []
for i in range(3):
    df = pd.read_csv(paths[i])
    seq_type = [seq_types[i]] * len(df["SEQUENCE"])
    df["seq_type"] = seq_type
    mics.append(df)
box_mics = pd.concat(mics)
box_mics = box_mics[colnames]
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"diffusion": "blue", "diffusion-classifier": "green", "diffusion-classifier-free":"pink"}
sns.boxplot(x = box_mics["seq_type"], y = box_mics["ec_predicted_MIC_μM"],
            hue = box_mics["seq_type"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free"])
plt.xlabel("Peptide type", size=1, fontweight='bold') #, fontweight='bold'
plt.ylabel("ec_predicted_MIC_μM",  size=15, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("svg/prot_props_figures/ec_predicted_MIC_μM.svg", bbox_inches='tight')
plt.close()


fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"diffusion": "blue", "diffusion-classifier": "green", "diffusion-classifier-free":"pink"}
sns.boxplot(x = box_mics["seq_type"], y = box_mics["sa_predicted_MIC_μM"],
            hue = box_mics["seq_type"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free"])
plt.xlabel("Peptide type", size=1, fontweight='bold') #, fontweight='bold'
plt.ylabel("sa_predicted_MIC_μM",  size=15, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("svg/prot_props_figures/sa_predicted_MIC_μM.svg", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"diffusion": "blue", "diffusion-classifier": "green",
           "diffusion-classifier-free":"pink", "True AMP":"yellowgreen", "Random Non-AMP":"coral"}
sns.boxplot(x = box_data["seq_type"], y = box_data["molecular_weight"],
            hue = box_data["seq_type"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free", "True AMP", "Random Non-AMP"])
plt.xlabel("Peptide type", size=1, fontweight='bold') #, fontweight='bold'
plt.ylabel("Molecular weight",  size=15, fontweight='bold')
plt.xticks(fontsize=10)
ax.xaxis.set_tick_params(rotation=20)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("svg/prot_props_figures/molecular_weight.svg", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
hue_color={"diffusion": "blue", "diffusion-classifier": "green",
           "diffusion-classifier-free":"pink", "True AMP":"yellowgreen", "Random Non-AMP":"coral"}
sns.boxplot(x = box_data["seq_type"], y = box_data["isoelectric_point"],
            hue = box_data["seq_type"], palette = hue_color, showfliers=0)
ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free", "True AMP", "Random Non-AMP"])
plt.xlabel("Peptide type", size=1, fontweight='bold') #, fontweight='bold'
plt.ylabel("Isoelectric point",  size=15, fontweight='bold')
plt.xticks(fontsize=10)
ax.xaxis.set_tick_params(rotation=20)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("svg/prot_props_figures/isoelectric_point.svg", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
# hue_color={"diffusion": "blue", "diffusion-classifier": "green", "diffusion-classifier-free":"pink"}
sns.boxplot(x = box_data["seq_type"], y = box_data["gravy"],
            hue = box_data["seq_type"], palette = hue_color, showfliers=0)
# ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free"])
ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free", "True AMP", "Random Non-AMP"])
plt.xlabel("Peptide type", size=1, fontweight='bold') #, fontweight='bold'
plt.ylabel("Gravy",  size=15, fontweight='bold')
plt.xticks(fontsize=10)
ax.xaxis.set_tick_params(rotation=20)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("svg/prot_props_figures/gravy.svg", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
# hue_color={"diffusion": "blue", "diffusion-classifier": "green", "diffusion-classifier-free":"pink"}
sns.boxplot(x = box_data["seq_type"], y = box_data["hydrophobic_moment"],
            hue = box_data["seq_type"], palette = hue_color, showfliers=0)
# ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free"])
ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free", "True AMP", "Random Non-AMP"])
plt.xlabel("Peptide type", size=1, fontweight='bold') #, fontweight='bold'
plt.ylabel("Hydrophobic Moment",  size=15, fontweight='bold')
plt.xticks(fontsize=10)
ax.xaxis.set_tick_params(rotation=20)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("svg/prot_props_figures/hydrophobic_moment.svg", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'sans Serif'
plt.rcParams['font.serif'] = ['Arial']
# hue_color={"diffusion": "blue", "diffusion-classifier": "green", "diffusion-classifier-free":"pink"}
sns.boxplot(x = box_data["seq_type"], y = box_data["charge"],
            hue = box_data["seq_type"], palette = hue_color, showfliers=0)
# ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free"])
ax.set_xticklabels(["diffusion", "diffusion-classifier", "diffusion-classifier-free", "True AMP", "Random Non-AMP"])
plt.xlabel("Peptide type", size=1, fontweight='bold') #, fontweight='bold'
plt.ylabel("Charge",  size=15, fontweight='bold')
plt.xticks(fontsize=10)
ax.xaxis.set_tick_params(rotation=20)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("svg/prot_props_figures/charge.svg", bbox_inches='tight')
plt.close()
print("smart")
