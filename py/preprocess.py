import os
import re
from collections import Counter

def DFS_file_search(dict_name):  #深度优先搜索文件夹，将txt文件返回到result_txt中
    # list.pop() list.append()这两个方法就可以实现栈维护功能
    stack = []
    result_txt = []
    stack.append(dict_name)
    while len(stack) != 0:  # 栈空代表所有目录均已完成访问
        temp_name = stack.pop()
        try:
            temp_name2 = os.listdir(temp_name)  #输出路径下的文件夹名
            for eve in temp_name2:
                stack.append(temp_name + "\\" + eve)  # 维持绝对路径的表达
        except NotADirectoryError:
            result_txt.append(temp_name)   #捕捉异常
    return result_txt

path_list = DFS_file_search(r".\nlp_corpus")  #防止路径中存在转义符
# path_list 为包含所有小说文件的路径列表
corpus = []
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip().replace("\u3000", "").replace("\t", "").replace(" ","") for line in file][3:-3]
        corpus += text
print(len(corpus))
print(corpus)
# corpus 存储语料库，其中以每一个自然段为一个分割
regex_str = ".*?([^\u4E00-\u9FA5]).*?"   #判断是否为中文
english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+' #unicode编码,防止出现乱码
symbol = []
for j in range(len(corpus)):
    corpus[j] = re.sub(english, "", corpus[j]) #将英文及符号消除
    symbol += re.findall(regex_str, corpus[j]) #讲中文放入symbol
count_ = Counter(symbol)#提出每个字出现的频率
count_symbol = count_.most_common()#转成dict格式
noise_symbol = []
for eve_tuple in count_symbol:
    if eve_tuple[1] < 500:
        noise_symbol.append(eve_tuple[0])
noise_number = 0
for line in corpus:
    for noise in noise_symbol:
        line.replace(noise, "")
        noise_number += 1
print("完成的噪声数据替换点：", noise_number)
print("替换的噪声符号：")
for i in range(len(noise_symbol)):
    print(noise_symbol[i], end=" ")#隔一个空格符
    if i % 50 == 0:
        print()
with open("预处理后的文本1.txt", "w", encoding="utf-8") as f:
    for line in corpus:
        if len(line) > 1:
            print(line, file=f)
