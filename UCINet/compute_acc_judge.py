f = open("judge_margin05result_margin08.txt","r",encoding="utf-8")
content=f.readlines()
lenth=len(content)
print(lenth)
acc=0

for each in content[:-1]:
    truth = int(each.split("\t")[2][6:].strip())
    prediction = int(each.split("\t")[3][5:].strip())
    if(truth == prediction):
        acc=acc+1
print(acc/lenth)

