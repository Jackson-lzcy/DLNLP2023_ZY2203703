import jieba
from collections import Counter
import math
import matplotlib.pyplot as plt

with open("预处理后的文本1.txt", "r", encoding="utf-8") as f:
    corpus = [eve.strip() for eve in f]
# print(corpus)
token = []
for para in corpus:
    token += jieba.lcut(para) #精确分词
# print(token)
token_num = len(token)
ct = Counter(token)
vocab1 = ct.most_common()
entropy_1gram = sum([-(eve[1]/token_num)*math.log2(eve[1]/token_num) for eve in vocab1])
print("词库总词数：", token_num, " ", "不同词的个数：", len(vocab1))
print("出现频率前10的1-gram词语：", vocab1[:10])
print("entropy_1gram:", entropy_1gram)

x = []
y = []
plt.rcParams["font.sans-serif"] = ["SimHei"]

for i in range(10):
   x.append(vocab1[i][0])
   y.append(vocab1[i][1])
plt.bar(x, y)
plt.title("词频前十的词及频数")
plt.xlabel("具体的词")
plt.ylabel("频数")
plt.show()