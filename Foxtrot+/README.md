Usage:

tsv2vw.py

vw -d train.vw -c -k -f model --loss_function logistic --passes 10 -p train_p.txt 
vw -t -d test.vw -i model -p test_p.txt

p2sub sampleSubmission.csv p.txt p.csv