def load_data(url):
    txt = requests.get(url).content
    txt = txt.decode('utf-8')
    # print(txt)
    x = []
    y = []
    txt = txt.split('\n')
    # print(txt)
    for line in txt[:-1]:
        xs = line.split('\t')
        y.append(float(xs[-1]))
        x1 = xs[:-1]
        for i in range(10):
            x1[i] = float(x1[i])
        x1 = [1] + x1
        # mul = np.sqrt(np.dot(x1, x1))
        # x1 = [i / mul for i in x1]
        x.append((x1))
        # print(x)
    x = np.array(x)
    y = np.array(y)
    return x, y