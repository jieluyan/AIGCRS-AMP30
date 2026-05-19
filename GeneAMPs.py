from DiffusionClassifier import *

new_seqs = []
# generate 1000 AMP sequences
generated_seqs = model.generate_AMPs(num=1000)
for seq in generated_seqs:
    if len(seq) > 4 and len(seq) < 31:
        new_seqs.append(seq)
id = ["geneAMP_%d" % (i+1) for i in range(len(new_seqs))]
df = pd.DataFrame({"ID": id, "SEQUENCE": new_seqs})
os.makedirs("./result/", exist_ok=True)
p_generated_seq = ("./result/Gene_Seqs.csv")
print("Tested AMP result will be saved to: ", p_generated_seq)
df.to_csv(p_generated_seq, header=True, index=False)