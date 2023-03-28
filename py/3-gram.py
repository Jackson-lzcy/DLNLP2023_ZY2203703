import jieba
from collections import Counter
import math
import matplotlib.pyplot as plt
import re

punctuation = '．'
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

with open("预处理后的文本1.txt", "r", encoding="utf-8") as f:
    # corpus = [eve.strip("\n").replace(".","") for eve in f]
    corpus = [removePunctuation(eve).strip("\n") for eve in f]

def combine3gram(cutword_list):
    if len(cutword_list) <= 2:
        return []
    res = []
    for i in range(len(cutword_list)-2):
        if cutword_list[i] == '.':
            continue
        else:
            res.append(cutword_list[i] + cutword_list[i+1] +" "+ cutword_list[i+2])
    return res
token_3gram = []
for para in corpus:
    cutword_list = jieba.lcut(para)
    token_3gram += combine3gram(cutword_list)

token_3gram_num = len(token_3gram)
ct3 = Counter(token_3gram)
vocab3 = ct3.most_common()#统计过后的

same_12same_word = [eve.split(" ")[0] for eve in token_3gram]
assert token_3gram_num == len(same_12same_word)
ct_12st = Counter(same_12same_word)
vocab_12st = dict(ct_12st.most_common()) #将list转换为dict
entropy_3gram = 0
for eve in vocab3:
    p_xyz = eve[1]/token_3gram_num
    first_12word = eve[0].split(" ")[0]
    entropy_3gram += -p_xyz*math.log2(eve[1]/vocab_12st[first_12word])
# del vocab3[5]
print("词库总词数：", token_3gram_num, " ", "不同词的个数：", len(vocab3))
print("出现频率前10的3-gram词语：", vocab3[:10])
print("entropy_3gram:", entropy_3gram)

x = []
y = []
plt.rcParams["font.sans-serif"] = ["SimHei"]

for i in range(10):
   x.append(vocab3[i][0])
   y.append(vocab3[i][1])
plt.bar(x, y)
plt.title("词频前十的词及频数")
plt.xlabel("具体的词")
plt.ylabel("频数")
plt.show()




