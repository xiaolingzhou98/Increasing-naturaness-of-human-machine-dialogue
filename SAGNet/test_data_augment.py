import pandas as pd
def test_augment(datapath,to_datapath):
#option
    data_copy = pd.DataFrame(columns=["domain","question_type","question","answer",'label','class'])
    # judge
    #data_copy = pd.DataFrame(columns=["domain","question_type","question","answer",'sen_label','classes'])
    data = pd.read_excel(datapath)
    for i in range(len(data)):
        for j in range(43):
        #option
            data_copy=data_copy.append({"domain":"0","question_type":"1","question":data.loc[i,"question"],"answer":data.loc[i,"answer"],"label":data.loc[i,"target"],"class":str(j)},ignore_index=True)
            #judge
            #data_copy=data_copy.append({"domain":"0","question_type":"0","question":data.loc[i,"question"],"answer":data.loc[i,"answer"],"sen_label":data.loc[i,"sen_label"],"classes":str(j)},ignore_index=True)
    data_copy.to_excel(to_datapath,index=False)
    
#test_augment("data/instances_dev_choose_multi_judge.xlsx","data/instances_dev_judge_class.xlsx")