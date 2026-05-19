import dill
import pandas as pd
def write_pkl(obj, file_path):
    with open(file_path, 'wb') as f:
        dill.dump(obj, f)
    return

def read_pkl(file_path):
    with open(file_path, "rb") as f:
        final_rs = dill.load(f)
    return final_rs

def seqs2fastas(seqs, path="./temp.fastas"):
    df = pd.DataFrame({"SEQUENCE": seqs})
    df["ID"] = df.index + 1
    df = df[["ID", "SEQUENCE"]]
    df.to_csv(path, header=True, index=False)
    print("save all sequences to: \n\t", path)
    return df

import os
from glob import glob
def getUnexistedName(file_path, file_type=".xlsx"):
    file_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    if not(os.path.isfile(file_path)):
        new_file = file_name
    else:
        name = file_name.split(file_type)[0]
        regex_str = name[:-1] + "*" + file_type
        regex_str = os.path.join(dir_name,regex_str)
        mylist = [f for f in glob(regex_str)]
        s = sorted(mylist)
        s = [os.path.basename(i) for i in s]
        s.remove(file_name)
        # print("file name:", file_name)
        # print(s)
        if len(s) > 0:
            nums = []
            for n in s:
                num_name = n.split(file_type)[0]
                # print(num_name)
                str_i = [str(i) for i in range(10)]
                str_num = num_name.split("-")[-1]
                a = [i in str_i for i in str_num]
                if len(str_num) == sum(a):
                    num = int(str_num)
                else:
                    s.remove(n)
                    continue
                nums.append(num)
            if len(nums) > 0:
                max_n = sorted(nums)[-1]
        if len(s) == 0:
            new_file = name + "-0" + file_type
        else:
            new_file = name + "-" + str(max_n+1) + file_type
    new_path = os.path.join(dir_name, new_file)
    return new_path

def getExistedLargestName(file_path, file_type=".xlsx"):
    file_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    if not(os.path.isfile(file_path)):
        new_file = file_name
        print("%s unexisted." % file_path)
    else:
        name = file_name.split(file_type)[0]
        regex_str = name[:-1] + "*" + file_type
        regex_str = os.path.join(dir_name,regex_str)
        mylist = [f for f in glob(regex_str)]
        s = sorted(mylist)
        s = [os.path.basename(i) for i in s]
        s.remove(file_name)
        if len(s) > 0:
            nums = []
            for n in s:
                num_name = n.split(file_type)[0]
                str_i = [str(i) for i in range(10)]
                str_num = num_name.split("-")[-1]
                a = [i in str_i for i in str_num]
                if len(str_num) == sum(a):
                    num = int(str_num)
                else:
                    s.remove(n)
                    continue
                    nums.append(num)
            if len(nums) > 0:
                max_n = sorted(nums)[-1]
        if len(s) == 0:
            new_file = file_name
        else:
            new_file = name + "-" + str(max_n) + file_type
    new_path = os.path.join(dir_name, new_file)
    return new_path