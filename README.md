1. After Downloaded this project, please visit goole drive by this link https://drive.google.com/drive/folders/1FTkL4aeZgI6k7zRjPMfczfwIV_rV3CME?usp=sharing, and download:
   1.1 feature folder, and put it to AMP_data folder, it should be: AMP_data/feature/amp_cai_5_30_amp_random_classifier_features.csv;
   1.2 checkpoints folder, and put it to the same folder (e.g., AIGCRS-AMP30 folder) of readme file.

2. Create the conda enviroment by code:
   conda env create -f environment.yml -n your_env_name
   
4. Activate the conda environment by code:
   conda activate your_env_name
   
5. cd to the AIGCRS-AMP30 folder, and generate the New AMP sequences by code:
   python GeneAMPs.py
     Then the generated sequences will be saved to: ./result/Gene_Seqs.csv
   
7. Test Given Seqnences by RFClassifier-AMP30 model by code:
   python RFTest.py
   The tested file are set to test_path = "./test/sample.csv"
   you can visit RFTest.py and change test_path to your path

8. If your sequence file for test is fasta file please covert it to csv file by code:
   python ./tools/fasta2csv.py ./test/test.fasta
   you can change the ./test/test.fasta to your fasta file path

