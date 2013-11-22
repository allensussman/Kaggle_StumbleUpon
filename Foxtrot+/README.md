Usage:

tsv2vw.py

vw -d train.vw -c -k -f model --loss_function logistic --passes 10 -p train_p.txt 

vw -t -d test.vw -i model -p test_p.txt

p2sub.py sampleSubmission_train.csv train_p.txt train_p.csv
p2sub.py sampleSubmission.csv test_p.txt test_p.csv
