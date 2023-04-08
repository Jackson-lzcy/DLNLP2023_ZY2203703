import numpy as np
import csv
import matplotlib.pylab as plt


def phi(Y, u, sigma):
    p = np.power((np.sqrt(2 * np.pi) * sigma), -1) * np.exp(-np.power(Y - u, 2) / (2 * sigma ** 2))
    return p

def E_step(Y, u1, u2,sigma1, sigma2, w1, w2):
    for k in range(i):
        y1 = phi(Y[k], u1, sigma1)
        y2 = phi(Y[k], u2, sigma2)
        gama1[k] = (w1 * y1) / (w1*y1 + w2*y2)
        gama2[k] = 1 - gama1[k]
    return gama1, gama2

def M_step(Y, gama, u, N):
    a = b = sum1 = 0
    for i in range(N):
        a += gama[i] * Y[i]
        b += gama[i]
    for i in range(N):
        sum1 += gama[i] * np.power((Y[i]-u), 2)

    return a/b, np.sqrt(sum1/b), b/N

if __name__ == '__main__':

    data0 = []
    data1 = []
    data = []  #身高数据
    gama1 = [0 for j in range(2000)] #数据对模型1响应度
    gama2 = [0 for j in range(2000)] #数据对模型2响应度
    w_m = []
    w_wm = []

    i = Iteration = 0
    u1 = 170
    u2 = 170
    w1 = 0.8
    w2 = 0.2
    sigma1 = sigma2 = 10
    real_u1 = real_u2 = 0

    with open("height_data.csv","r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] == 'height':
                continue
            data0.append(line)
            i = i + 1
    data = sum(data0, data1)
    for j in range(len(data)):
        data[j] = float(data[j])
    while True:
        gama1, gama2 = E_step(data, u1, u2, sigma1, sigma2, w1, w2)
        p = w1
        q = w2
        w_m.append(w1)
        w_wm.append(w2)
        u1, sigma1, w1 = M_step(data, gama1, u1, len(data))
        u2, sigma2, w2 = M_step(data, gama2, u2, len(data))
        Iteration += 1
        if (abs(w1-p) < 0.0000000000001) and (abs(w2-q) < 0.0000000000001):
            break
    for  i in range(len(data)):
        if i <500:
            real_u2 += data[i]
        else:
            real_u1 += data[i]
    real_u1 = real_u1/1500.0
    real_u2 = real_u2/500.0
    res1 = abs(u1 - real_u1)
    res2 = abs(u2 - real_u2)


    print("预测男生人数占比w1=%f" % w1, "预测女生人数占比w2=%f" % w2)
    print("预测男生身高均值u1=%f" % u1, "预测女生身高均值u2=%f" % u2)
    print("预测男生身高标准差sigma1=%f" % sigma1, "预测女生身高标准差sigma2=%f" % sigma2)
    print("迭代次数=%d" % Iteration)
    print("实际男生身高均值real_u1=%f" % real_u1, "实际女生身高均值real_u2=%f" % real_u2)
    print("男生均值之差=%f" % res1, "女生均值之差=%f" % res2)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure(1)
    plt.scatter(data,gama1)
    plt.scatter(data,gama2)
    plt.axis([150,195,0,1.01])
    plt.title('Responsiveness of two distributions to mixed height')
    plt.xlabel('Height(cm)')
    plt.ylabel('Responsiveness')

    plt.figure(2)
    plt.hist(data, bins=20, range=[150,195], color='#75C7B1')
    plt.title('Mixed Height Distributions')
    plt.xlabel('Height(cm)')
    plt.ylabel('Number')

    g1 = np.arange(148, 190, 0.1)
    h1 = phi(g1,u1,sigma1)
    plt.plot(g1,h1*3305,'b-')

    g2 = np.arange(148, 190, 0.1)
    h2 = phi(g2, u2, sigma2)
    plt.plot(g2, h2*1105, 'r-')

    plt.figure(3)
    plt.title("男女人数占比权重随迭代次数的变化")
    plt.xlabel("Iteration")
    plt.ylabel("权重占比w")
    plt.plot(range(Iteration), w_m, 'b-')
    plt.plot(range(Iteration), w_wm, 'r-')

    plt.show()


    # h = np.array(data)
    # h1 = pd.Series(h).hist(bins=150)
    # plt.plot(h1)
    # plt.title("身高分布")
    # plt.show()

    # for k in range(i):
    #     y1 = phi(data[k],u1,sigma1)
    #     y2 = phi(data[k],u2,sigma2)
    #     gama1[k] = (w1 * y1) / (w1*y1 + w2*y2)
    #     gama2[k] = (w2 * y2) / np.sum([w1*y1,w2*y2])


# b=phi(194,u1,sigma1)
# c= np.power((np.sqrt(2 * np.pi) * sigma1), -1) * np.exp(-np.power(190 - u1, 2) / 2 * (sigma1 ** 2))
# d = normal_distribution(194,u1,sigma1)
# print(b,d)
# q= np.power((np.sqrt(2 * np.pi) * sigma1), -1)
# p= 1/(np.sqrt(2*np.pi)* sigma1)
# m= np.exp(-np.power(190 - u1, 2) / 2 * (sigma1 ** 2))
# n= np.exp(-1*((190-u1)**2)/(2*(sigma1**2)))
# print(q,p)
# print(m,n)
# o=-225/2*10**2
# l=np.exp(-225/(2*10**2))
# print(o,l)
# def normal_distribution(x, mean, sigma):
#     return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)* sigma)


