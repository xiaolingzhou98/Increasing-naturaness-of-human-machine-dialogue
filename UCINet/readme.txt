Requirement:

Python=3.6, Tensorflow=1.3.1, pyltp=0.2.1 and numpy=1.16.2

Train and Test

Train the model,you need to download the pretrained model ltp_model and download the bert model BERT if you want to use bert pretrained vector.
Run the following command to train or evaluate the UCINet respectively.
run choose metric:
python metric_choose_train.py  
python metric_choose_test.py
run choose base using word2vec
python choose_train.py  
python choose_test.py
run choose base using glove:
python choose_train_glove.py  
python choose_test_glove.py

run judge metric
python metric_judge_train_glove.py  
python metric_judge_test_glove.py
run judge base using glove
python judge_train_glove.py  
python judge_test_glove.py

calculate acc
1 choose base£ºrun compute_acc.py
2 choose metric£ºrun two_comute_acc.py
3 judge£ºrun compute_acc_judge.py

