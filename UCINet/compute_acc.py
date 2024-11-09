def acc(data_path):
    f = open(data_path,"r",encoding="utf-8")
    content=f.readlines()
    lenth=len(content)
    print(lenth)
    acc=0
    sum=0
    correct = 0
    total = 0
    for each in content:
        #sum=0
    #    avg=0
        begin = each.index("options")+8
        end = each.index("]")
        options = each[begin:end]
        option = []
        for op in options.split(" "):
            if(op!=""):
                option.append(op)
        len_option = len(option)
        total = total+len_option
        #print(option)
        #print(len_option)
        if(len_option<=5):
            #total=total+1
            begin_label = each.index("label:")+6
            end_label = each.index("]",begin_label)
            label = each[begin_label:end_label].strip("[")
            label_list = label.split(",")
            pre_begin = each.index("prediction_ls")+13
            pre_end = each.index("\n",pre_begin)
            pre = each[pre_begin:pre_end].split(" ")
            label1=[]
            pre1 = []
            for la in label_list:
                if(la!=""):
                    label1.append(int(la))
            for pr in pre:
                if(pr!=""):
                    pre1.append(int(pr))
            #print(label1)
            #print(pre1)
            #if(label1[0:len_option] == pre1[0:len_option]):
            #    sum=sum+1
            #if(len(label1)==len(pre1)==len_option):
            #    total = total+len_option
            for i in range(len_option):
                if(label1[i] == pre1[i]):
                    sum=sum+1
            #                correct = correct+1
             #   avg = sum/len_option
             #   acc=acc+avg
    
    accuracy = sum/total
    return(accuracy)
    #print(accuracy)

print(acc("/home/nlp/Desktop/AntNet-rverseQA-master/result_xiaorong_add1.txt"))
