from RFtune import *
# put all the test files to the list below
test_path = "./test/sample.csv"
y_pre, prob = RFDevelopAndTest(test_path=test_path)
p_csv = test_path
if ".fasta" in test_path:
    p_csv = test_path.replace(".fasta", ".csv")
df = pd.read_csv(test_path)
df["AMP_class"] = y_pre
df["AMP_prob"] = prob
s = pd.Series(df["SEQUENCE"])
lens = s.str.len()
df["Length"] = lens
p_generated_seq = p_csv.replace(".csv", "_RFPre_AMP.csv")
print("Tested AMP result will be saved to: ", p_generated_seq)
df.to_csv(p_generated_seq, header=True, index=False)