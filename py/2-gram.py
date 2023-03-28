import jieba
from collections import Counter
import math
import matplotlib.pyplot as plt

with open("预处理后的文本1.txt", "r", encoding="utf-8") as f:
    corpus = [eve.strip() for eve in f]

def combine2gram(cutword_list):
    if len(cutword_list) == 1:
        return []
    res = []
    for i in range(len(cutword_list)-1):
        res.append(cutword_list[i] + " "+ cutword_list[i+1])
    return res
token_2gram = []
for para in corpus:
    cutword_list = jieba.lcut(para)
    token_2gram += combine2gram(cutword_list)
# 2-gram的频率统计
token_2gram_num = len(token_2gram)
ct2 = Counter(token_2gram)
vocab2 = ct2.most_common()
# print(vocab2[:20])
# 2-gram相同句首的频率统计
same_1st_word = [eve.split(" ")[0] for eve in token_2gram]
assert token_2gram_num == len(same_1st_word)
ct_1st = Counter(same_1st_word)
vocab_1st = dict(ct_1st.most_common()) #将list转换为dict
entropy_2gram = 0
for eve in vocab2:
    p_xy = eve[1]/token_2gram_num
    first_word = eve[0].split(" ")[0]
    # p_y = eve[1]/vocab_1st[first_word]
    entropy_2gram += -p_xy*math.log2(eve[1]/vocab_1st[first_word])
print("词库总词数：", token_2gram_num, " ", "不同词的个数：", len(vocab2))
print("出现频率前10的2-gram词语：", vocab2[:10])
print("entropy_2gram:", entropy_2gram)

x = []
y = []
plt.rcParams["font.sans-serif"] = ["SimHei"]

for i in range(10):
   x.append(vocab2[i][0])
   y.append(vocab2[i][1])
plt.bar(x, y)
plt.title("词频前十的词及频数")
plt.xlabel("具体的词")
plt.ylabel("频数")
plt.show()
