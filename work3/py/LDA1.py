import math
import jieba
import os  # 用于处理文件路径
import random
import numpy as np


def read_novel(path):  # 读取语料内容
    content = []
    names = os.listdir(path)
    for name in names:
        con_temp = []
        novel_name = path + '\\' + name
        with open(novel_name, 'r', encoding='ANSI') as f:
            con = f.read()
            con = content_deal(con)
            con = jieba.lcut(con)  # 结巴分词
            con_list = list(con)
            pos = int(len(con) // 13)
            for i in range(13):
                con_temp = con_temp + con_list[i * pos: i * pos + 1000]
            content.append(con_temp)
        f.close()
    return content, names

def read_novel1(path):  # 读取语料内容——测试集
    content = []
    names = os.listdir(path)
    for name in names:
            con_temp = []
            novel_name = path + '\\' + name
            with open(novel_name, 'r', encoding='ANSI') as f:
                con = f.read()
                con = content_deal(con)
                con = jieba.lcut(con)  # 结巴分词
                con_list = list(con)
                pos = int(len(con)//13) ####16篇文章，分词后，每篇均匀选取13个500词段落进行建模
                for i in range(13):
                    con_temp = con_list[i*pos+500: i*pos+1000] #测试集
                    content.append(con_temp)
            f.close()
    return content, names

def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b',' ']
    for a in ad:
        content = content.replace(a, '')
    return content

def read_word_novel(path):  # 读取语料内容
    content = []
    names = os.listdir(path)
    for name in names:
        con_temp = []
        novel_name = path + '\\' + name
        content1 = []
        with open(novel_name, 'r', encoding='ANSI') as f:
            con = f.read()
            con = content_deal(con)
            for word in con:
                con = list(word)
                for eve in con:
                    content1.append(eve)
        pos = int(len(content1)//13)
        for i in range(13):
            con_temp = con_temp + content1[i*pos: i*pos+1000]
        content.append(con_temp)
        f.close()
    return content, names

def read_word_novel1(path):  # 读取语料内容——测试集
    content = []
    content2 = []
    names = os.listdir(path)
    for name in names:
            con_temp = []
            novel_name = path + '\\' + name
            with open(novel_name, 'r', encoding='ANSI') as f:
                con = f.read()
                con = content_deal(con)
                for word in con:
                    con = list(word)
                    for eve in con:
                        content2.append(eve)
                con_list = list(content2)
                pos = int(len(content2)//13) ####16篇文章，分词后，每篇均匀选取13个500词段落进行建模
                for i in range(13):
                    con_temp = con_list[i*pos+500:i*pos+1000] #测试集
                    content.append(con_temp)
            f.close()
    return content, names

def Topic_num(topic_num):
    Topic_word_fre = {}
    for i in range(topic_num):
        Topic_word_fre[i] = Topic_word_fre.get(i, {})
    return Topic_word_fre


def train(Doc_txt,topic_nums):
    Docword_from_topic = []  # 每篇文章中的每个词来自哪个topic

    Topic_all_word = {}  # 每个topic有多少词
    Topic_word_fre = Topic_num(topic_nums)  # 存每个topic的词频

    Doc_all_word = []  # 每篇文章中有多少个词
    Doc_word_from_topic = []  # 每篇文章有多少各个topic的词

    for data in data_txt:
        topic = []  # 存每个词的topic
        docfre = {}
        for word in data:
            a = random.randint(0, topic_nums - 1)  # 为每个单词赋予一个随机初始topic
            topic.append(a)
            if '\u4e00' <= word <= '\u9fa5':
                Topic_all_word[a] = Topic_all_word.get(a, 0) + 1  # 统计每个topic总词数
                docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章来自每个topic的词频
                Topic_word_fre[a][word] = Topic_word_fre[a].get(word, 0) + 1  # 统计每个topic的词频
        for i in range(topic_nums):
            docfre[i] = docfre.get(i, 0)
        Docword_from_topic.append(topic)
        docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())  # 排序让每个topic对齐
        Doc_word_from_topic.append(docfre)
        Doc_all_word.append(sum(docfre))  # 统计每篇文章的总词数

    Topic_all_word = list(dict(sorted(Topic_all_word.items(), key=lambda x: x[0], reverse=False)).values())

    Doc_word_from_topic = np.array(Doc_word_from_topic)  # 转为array方便后续计算
    Topic_all_word = np.array(Topic_all_word)  # 转为array方便后续计算
    Doc_all_word = np.array(Doc_all_word)  # 转为array方便后续计算
    print("每篇文档来自每个topic的词数量:", Doc_word_from_topic)
    print("每篇主题的总词数:", Topic_all_word)
    print("每篇文章的总词数:", Doc_all_word)

    Doc_pro = []  # 每篇文章的主题概率分布

    Doc_pronew = []  # 记录每次迭代后每个topic被选中的新概率
    for i in range(len(data_txt)):
        Doc_pro1 = []
        for j in range(topic_nums):
            doc = np.divide(Doc_word_from_topic[i][j], Doc_all_word[i])
            Doc_pro1.append(doc)
        Doc_pro.append(Doc_pro1)
    Doc_pro = np.array(Doc_pro)
    print(Doc_pro)
    stop = 0  # 迭代停止标志
    loopcount = 1  # 迭代次数
    while stop == 0:
        i = 0
        for data in data_txt:
            top = Docword_from_topic[i]  # 文章中每个词属于的topic集合
            for w in range(len(data)):
                word = data[w]
                pro = []
                topfre = []
                if '\u4e00' <= word <= '\u9fa5':
                    for j in range(topic_nums):
                        topfre.append(Topic_word_fre[j].get(word, 0))  # 读取该词语在每个topic中出现的频数
                    pro = Doc_pro[i] * topfre / Topic_all_word  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                    m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的,取下标
                    Doc_word_from_topic[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                    Doc_word_from_topic[i][m] += 1
                    Topic_all_word[top[w]] -= 1  # 更新每个topic的总词数
                    Topic_all_word[m] += 1
                    Topic_word_fre[top[w]][word] = Topic_word_fre[top[w]].get(word, 0) - 1  # 更新每个topic该词的频数
                    Topic_word_fre[m][word] = Topic_word_fre[m].get(word, 0) + 1
                    top[w] = m
            Docword_from_topic[i] = top
            i += 1
        print(Doc_word_from_topic, 'new')
        print(Topic_all_word, 'new')
        if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
            for i in range(len(data_txt)):
                Doc_pro2 = []
                for j in range(topic_nums):
                    doc = np.divide(Doc_word_from_topic[i][j], Doc_all_word[i])
                    Doc_pro2.append(doc)
                Doc_pronew.append(Doc_pro2)
            Doc_pronew = np.array(Doc_pronew)
        else:
            for i in range(len(data_txt)):
                for j in range(topic_nums):
                    doc = np.divide(Doc_word_from_topic[i][j], Doc_all_word[i])
                    Doc_pronew[i][j] = doc
        print(Doc_pro)
        print(Doc_pronew)
        if (Doc_pronew == Doc_pro).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为模型已经训练完毕
            stop = 1
        else:
            Doc_pro = Doc_pronew.copy()
        loopcount += 1
    print(Doc_pronew)  # 输出最终训练的到的每篇文章选中各个topic的概率
    print(loopcount)  # 输出迭代次数
    # print(Doc_pro0)
    # print(Doc_pronew0)
    print('模型训练完毕！')
    return Topic_all_word, Topic_word_fre, Doc_pro, loopcount


def test(train_txt, test_txt, topic_nums, Topic_all_word, Topic_fre, Doc_pro):
    Doc_count_test = []  # 每篇文章中有多少个词
    Doc_fre_test = []  # 每篇文章有多少各个topic的词
    Topic_All_test = []  # 每篇文章中的每个词来自哪个topic
    for data in test_txt:
        topic = []
        docfre = {}
        for word in data:
            a = random.randint(0, topic_nums - 1)  # 为每个单词赋予一个随机初始topic
            topic.append(a)
            if '\u4e00' <= word <= '\u9fa5':
                docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的主题词频
        for i in range(topic_nums):
            docfre[i] = docfre.get(i, 0)
        Topic_All_test.append(topic)
        docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
        Doc_fre_test.append(docfre)
        Doc_count_test.append(sum(docfre))  # 统计每篇文章的总词数

    # print(Topic_All[0])
    Doc_fre_test = np.array(Doc_fre_test)
    Doc_count_test = np.array(Doc_count_test)
    print(Doc_fre_test)
    print(Doc_count_test)
    Doc_pro_test = []  # 每个topic被选中的概率
    Doc_pronew_test = []  # 记录每次迭代后每个topic被选中的新概率
    for i in range(len(test_txt)):
        Doc_pro_test1 = []
        for j in range(topic_nums):
            doc = np.divide(Doc_fre_test[i][j], Doc_count_test[i])
            Doc_pro_test1.append(doc)
        Doc_pro_test.append(Doc_pro_test1)
    Doc_pro_test = np.array(Doc_pro_test)
    print(Doc_pro_test)
    stop = 0  # 迭代停止标志
    loopcount = 1  # 迭代次数
    while stop == 0:
        i = 0
        for data in test_txt:
            top = Topic_All_test[i]
            for w in range(len(data)):
                word = data[w]
                pro = []
                topfre = []
                if '\u4e00' <= word <= '\u9fa5':
                    for j in range(topic_nums):
                        topfre.append(Topic_fre[j].get(word, 0))  # 读取该词语在每个topic中出现的频数
                    pro = Doc_pro_test[i] * topfre / Topic_all_word  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                    m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                    Doc_fre_test[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                    Doc_fre_test[i][m] += 1
                    top[w] = m
            Topic_All_test[i] = top
            i += 1
        print(Doc_fre_test, 'new')
        if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
            for i in range(len(test_txt)):
                Doc_pro2_test = []
                for j in range(topic_nums):
                    doc = np.divide(Doc_fre_test[i][j], Doc_count_test[i])
                    Doc_pro2_test.append(doc)
                Doc_pronew_test.append(Doc_pro2_test)
            Doc_pronew_test = np.array(Doc_pronew_test)
        else:
            for i in range(len(test_txt)):
                for j in range(topic_nums):
                    doc = np.divide(Doc_fre_test[i][j], Doc_count_test[i])
                    Doc_pronew_test[i][j] = doc
        print(Doc_pro_test)
        print(Doc_pronew_test)
        if (Doc_pronew_test == Doc_pro_test).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为训练集已分类完毕
            stop = 1
        else:
            Doc_pro_test = Doc_pronew_test.copy()
        loopcount += 1
        t = ((Doc_pronew_test == Doc_pro_test).all())
        print("分布是否相同：", t)
        print("迭代次数:", loopcount)
        if(loopcount == 20):
            stop = 1
    print(Doc_pro)
    print(Doc_pro_test)
    print(loopcount)
    print('测试集测试完毕！')
    result = []
    for k in range(len(test_txt)):
        pro = []
        for i in range(len(train_txt)):
            dis = 0
            for j in range(topic_nums):
                dis += (Doc_pro[i][j] - Doc_pro_test[k][j]) ** 2  # 计算欧式距离
            pro.append(dis)
        m = pro.index(min(pro))
        print(pro)
        result.append(m)  #小说的索引
    print(files)
    print(result)
    return result, loopcount

if __name__ == '__main__':

    mode = 0 #0代表以字为单位，1代表以词为单位
    right = wrong = a = 0
    topic_nums = 300  #主题数量
    path = "./nlp_corpus1"
    if mode == 0:
        [data_txt, files] = read_word_novel(path) #读部分语料库内容
        [test_txt, files] = read_word_novel1(path) #读部分语料库做测试
    else:
        [data_txt, files] = read_novel(path)  # 读部分语料库内容
        [test_txt, files] = read_novel1(path)  # 读部分语料库做测试

    [Topic_all_word, Topic_word_fre, Doc_pro, loop_train] = train(data_txt, topic_nums)
    [result, loop_test] = test(data_txt, test_txt, topic_nums, Topic_all_word, Topic_word_fre, Doc_pro)

    for j in range(len(test_txt)):
        a = (j // 13)
        if result[j] == a:
            right += 1
        else:
            wrong += 1
    assert (right + wrong == len(test_txt))
    print("正确分类的数量:", right)
    print("错误分类的数量:", wrong)
    print("正确分类率", float(right)/len(test_txt))
    print("训练迭代次数", loop_train)
    print("测试迭代次数", loop_test)





