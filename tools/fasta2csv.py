from Bio import SeqIO
import sys
from pathlib import Path
import pandas as pd

unnatural_amino_acids = ["B", "J", "O", "U", "Z", "X"]

def readFastaYan(fasta_file):
    seqs = []
    lens = []
    names = []
    records =[]
    for record in SeqIO.parse(fasta_file, "fasta"):
        # read Sequences
        seqs.append(record.seq)
        # get sequence length
        lens.append(len(record.seq))
        # get sequence name record.id = "P61542|ori"
        name = record.id.split("|")[0]
        names.append(name)
        record.id = name
        records.append(record)
    return names, seqs, lens, records

def geneFastasFromFastaFile(input_filepath: Path):
    seqs = []
    lens = []
    names = []
    records =[]
    names, seqs, lens, records = readFastaYan(input_filepath)
    fastas_file = str(input_filepath).replace(".fasta", ".csv")
    fastas = pd.DataFrame({"ID": names, "SEQUENCE": seqs})
    fastas.to_csv(fastas_file, header=True, index=False)
    print("fasta file was Converted to fastas which is a csv file with ID and SEQUENCE columns: \n\t", fastas_file)
    return fastas, fastas_file

import sys, os


def convert(input,output):
    """
    Purpose: To convert a .fasta file of >= 1 sequence(s) into a .csv file, with
    two columns, one containing the headline identifier, the other containing the
    sequence.
    Parameters: the input .fasta file path, a string, and the desired .csv output
    path, another string.
    Return: the output path, a string.
    """
    if not os.path.exists(input):
        raise IOError(errno.ENOENT, 'No such file', input)

    # Read in Fasta
    fasta = open(input, 'r')
    fasta_lines = fasta.readlines()
    seq = {}
    seqs = []
    c = 0
    for line in fasta_lines:

        if line[0] == ">": # head line with description
            seqs += [seq] # adding dicitionary to broader list
            c +=1
            seq_id = "p" + str(c)
            seq_local = {}
            # seq_head = line.strip(">\n")
            seq_local["seq_type"] = seq_id # identifier
            seq_local["seq"] = "" # actual sequence
            seq = seq_local

        else: # sequence line
            seq["seq"] += line.strip("\n")

    fasta.close()

    # Convert fasta to csv
    seqs.pop(0) # removing first (empty) item in seqs list i.e. fencepost
    csv_lines = ["Properties, Sequence\n"]
    for seq in seqs:
        not_conatin_unnatural_aa_flag = not any(aa in seq["seq"] for aa in unnatural_amino_acids)
        if not_conatin_unnatural_aa_flag:
            csv_line = seq["seq_type"] + "," + seq["seq"] + "\n"
            csv_lines += csv_line

    # Output csv file
    csv = open(output, 'w')
    csv.writelines(csv_lines)
    csv.close()
    return output

def convertWithAttributes(input,output):
    """
    Purpose: To convert a .fasta file of >= 1 sequence(s) into a .csv file, with
    n >= 2 columns, n-1 of which contain attributes of the sequence listed in the
    headline identifier, and 1 of which contains the actual sequence.
    Parameters: the input .fasta file path, a string, and the desired .csv output
    path, another string.
    Return: the output path, a string.
    """
    if not os.path.exists(input):
        raise IOError(errno.ENOENT, 'No such file', input)

    # Read in Fasta
    # for fasta in chain(*(open(input, 'r') for filepath in input_dirpath.iterdir())):
    fasta = open(input, 'r')
    fasta_lines = fasta.readlines()
    seq = {}
    seqs = []

    for line in fasta_lines:

        if line[0] == ">": # head line with description
            seqs += [seq] # adding dicitionary to broader list
            seq_local = {}
            seq_head = line.strip(">\n").split("|") # seperating the head's attributes
            seq_local["seq_type_list"] = seq_head # identifier
            seq_local["seq"] = "" # actual sequence
            seq = seq_local

        else: # sequence line
            seq["seq"] += line.strip("\n")

    fasta.close()

    # Convert fasta to csv
    seqs.pop(0) # removing first (empty) item in seqs list i.e. fencepost
    csv_lines = []
    for seq in seqs:
        csv_line = ""
        for type in seq["seq_type_list"]:
            csv_line += (type + ",")

        csv_lines += (csv_line + "\n")

    # Output csv file
    csv = open(output, 'w')
    csv.writelines(csv_lines)
    csv.close()
    return output

if __name__=="__main__" :
    geneFastasFromFastaFile(
        input_filepath=Path(sys.argv[1])
    )
    
    # convert("/home/yanjielu/acp-design/Gene-EC/data/AMP_data/addition/uniprotkb_length_5_TO_50_2024_02_27.fasta",
    #         "/home/yanjielu/acp-design/Gene-EC/data/AMP_data/addition/uniprotkb_length_5_TO_50_2024_02_27.csv")
    
