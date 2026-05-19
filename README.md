Following the step 1 to 3, AIGCRS-AMP30 can be installed. 

With the step 4, novel AMP canidate sequences can be generated. 
With the step 5, AMP probability score can be predicted. 

And then the MIC values of E. coli and S. aureus and HC50 values can be predicted by the other 2 projects (step 7 & 8).


1. After downloaded this project, please visit goole drive by this link https://drive.google.com/drive/folders/1FTkL4aeZgI6k7zRjPMfczfwIV_rV3CME?usp=sharing, and download:

   1.1 cd to the AIGCRS-AMP30 folder, and check the 4 downloaded folders, their names are: checkpoints, model, ifeature, feature.
   
   1.2 put the first 3 folders: checkpoints, model, ifeature to the current folder (AIGCRS-AMP30 folder).
   
   1.3 put the feature folder to AMP_data folder, it should be: "AMP_data/feature/amp_cai_5_30_amp_random_classifier_features.csv".
    

2. Create the conda enviroment by code:

   conda env create -f environment.yml -n your_env_name

   
3. Activate the conda environment by code:

   conda activate your_env_name

   
4. Generate the New AMP sequences by code:

   python GeneAMPs.py

      Then the generated sequences will be saved to: ./result/Gene_Seqs.csv

   
5. Test Given Seqnences by RFClassifier-AMP30 model by code:

   python RFTest.py

      The tested file are set to test_path = "./test/sample.csv".
      you can visit RFTest.py and change test_path to your path

6. If your sequence file for test is fasta file please covert it to csv file by code:

    python ./tools/fasta2csv.py ./test/test.fasta
   
      you can change the ./test/test.fasta to your fasta file path

7. For MIC values prediction of E. coli and S. aureus, please install by this project: https://github.com/janecai0714/AMP_regression_EC_SA, and predict follow its introduction.

    
8. For 50% haemolytic concentration (HC50) value prediction, please visit: https://app.cbbio.online/hemopep/home, and predict follow its introduction.
