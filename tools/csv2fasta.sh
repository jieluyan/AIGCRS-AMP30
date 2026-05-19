#!/bin/bash
ROOT_PATH="/home/yanjielu/AMP-Gene/AMP_data"
# name="PeptideAtlas_5_30_del_ampCai_random"
name="amp_cai_5_30"
infile=$ROOT_PATH/$name.csv
temp=$ROOT_PATH/temp.fasta
outfile=$ROOT_PATH/$name.fasta
awk -F, '{print ">"$1"\n"$2}' $infile > $temp
tail -n +3 $temp > $outfile
rm $temp
