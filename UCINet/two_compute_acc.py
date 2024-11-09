
f = open("result_margin0808.txt","r",encoding="utf-8")
content=f.readlines()
lenth=len(content)
acc=0
sum=0
total = 0
for each in content:
    

    begin = each.index("option")+8
    end = each.index("]")
    option = each[begin:end]
    len_option = len(option.split(","))
    #total = total + 3*len_option
    total = total + 3
    print(option.split(","))
    begin_label1 = each.index("label1:")+7
    end_label1 = each.index("]",begin_label1)
    begin_label2 = each.index("label2:")+7
    end_label2 = each.index("]",begin_label2)
    begin_label3 = each.index("label3:")+7
    end_label3 = each.index("]",begin_label3)
    label1 = each[begin_label1:end_label1].strip("[")
    label1_list = label1.split(" ")
    label2 = each[begin_label2:end_label2].strip("[")
    label2_list = label2.split(" ")
    label3 = each[begin_label3:end_label3].strip("[")
    label3_list = label3.split(" ")
    pre_begin1 = each.index("pre_x")+5
    pre_end1 = each.index("a",pre_begin1)
    pre1= each[pre_begin1:pre_end1].strip().split(" ")
    pre_begin2 = each.index("pre_xp")+7
    pre_end2 = each.index("a",pre_begin2)
    pre2= each[pre_begin2:pre_end2].strip().split(" ")
    pre_begin3 = each.index("pre_xn")+7
    pre_end3 = each.index("\n",pre_begin3)
    pre3= each[pre_begin3:pre_end3].strip().split(" ")
    label_one=[]
    pre_one = []
    label_two=[]
    pre_two = []
    label_three=[]
    pre_three = []
    for each in label1_list:
        if(each!=""):
            label_one.append(int(each.strip(",")))
    for each in pre1:
        #print(each)
        if(each!=""):
            pre_one.append(int(each))
    for each in label2_list:
        if(each!=""):
            label_two.append(int(each.strip(",")))
    for each in pre2:
        if(each!=""):
            pre_two.append(int(each))
    for each in label3_list:
        if(each!=""):
            label_three.append(int(each.strip(",")))
    for each in pre3:
        if(each!=""):
            pre_three.append(int(each))
    #for i in range(len_option):
    #    if(label_one[i] == pre_one[i] and label_one[i]!=3):
    #        sum=sum+1
    #for i in range(len_option):
    #    if(label_two[i] == pre_two[i] and label_one[i]!=3):
    #        sum=sum+1
    #for i in range(len_option):
    #    if(label_three[i] == pre_three[i] and label_one[i]!=3):
    #        sum=sum+1
    if(label_one[0:len_option] == pre_one[0:len_option]):
        sum=sum+1
    if(label_two[0:len_option] == pre_two[0:len_option]):
        sum=sum+1
    if(label_three[0:len_option] == pre_three[0:len_option]):
        sum=sum+1
accuracy = sum/total*100
print(accuracy)

