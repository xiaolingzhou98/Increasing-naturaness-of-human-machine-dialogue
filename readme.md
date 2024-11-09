# Increasing naturaness of human-machine dialogue.
 
This project includes two network for reverse QA, namely, UCINet for the inference of  user's choices on options in machine-raised questions, and SAGNet for automatic answer generation.<br>

Setups
-------  
The requiring environment is as bellow:<br>
* python 3.8<br>
* tensorflow 1.15.0<br>
* pyltp 0.2.1<br>
* numpy 1.16.2<br>


For SAGNet, Train the model,you need to download the pretrained model ltp_model.
-------  
Here are examples for training SAGNet:<br>
For Mdata:<br>
run sag_choose_train.py<br>
run sag_choose_test.py<br>
For Tdata:<br>
run judge_train_.py<br>
run judge_test_.py<br>


Train the model,you need to download the pretrained model ltp_model and download the bert model BERT if you want to use bert pretrained vector. Run the following command to train or evaluate the UCINet respectively.
-------  
run choose metric:<br>
python metric_choose_train.py<br>
python metric_choose_test.py<br>

run judge metric<br>
python metric_judge_train_glove.py<br>  
python metric_judge_test_glove.py<br>
