import numpy as np
import os

def add_bias(X):
    n = X.shape[0]
    ones = np.ones((n, 1))
    return np.hstack((ones, X))

def grad(X, y, th):
    n = X.shape[0]
    s = X @ th
    yz = y * s
    ez = np.exp(np.clip(yz, -500, 500))
    den = 1 + ez
    m = 1 / den
    g = (- X.T @ (y * m)) / n
    return g

def train_lr(X, y, lr=0.1, ep=1000):
    d = X.shape[1]
    th = np.zeros(d)
    for _ in range(ep):
        g = grad(X, y, th)
        th = th - lr * g
    return th

def train_fair(X, y, z, c, lr=0.1, ep=1000):
    n, d = X.shape
    th = np.zeros(d)
    zc = z - np.mean(z)
    v = (X.T @ zc) / n
    v2 = np.dot(v, v)
    for _ in range(ep):
        g = grad(X, y, th)
        th_temp = th - lr * g
        val = np.dot(th_temp, v)
        if val > c:
            corr = (val - c) / v2
            th_temp = th_temp - corr * v
        elif val < -c:
            corr = (val + c) / v2
            th_temp = th_temp - corr * v
        th = th_temp
    return th

def pred(X, th):
    s = X @ th
    return np.where(s >= 0, 1, -1)

def acc(y, yhat):
    return np.mean(y == yhat)

def prule(yhat, z):
    y1 = yhat[z == 1]
    y0 = yhat[z == 0]
    p1 = np.mean(y1 == 1)
    p0 = np.mean(y0 == 1)
    if p0 == 0:
        return 0
    r1 = p1 / p0
    r2 = p0 / p1
    return min(r1, r2) * 100

def load_adult_dataset():

    def read_file(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                row = [x.strip() for x in line.split(",")]
                if len(row) != 15:
                    continue
                data.append(row)
        return np.array(data)

    train_raw = read_file(os.path.join("adult", "adult.data"))
    test_raw = read_file(os.path.join("adult", "adult.test"))

    if test_raw[0][0] == "|1x3 Cross validator":
        test_raw = test_raw[1:]

    train_raw = train_raw[~np.any(train_raw == "?", axis=1)]
    test_raw = test_raw[~np.any(test_raw == "?", axis=1)]

    y_train = np.where(train_raw[:, -1] == ">50K", 1, -1)
    y_test = np.where(np.char.replace(test_raw[:, -1], ".", "") == ">50K", 1, -1)

    z_train = np.where(train_raw[:, 9] == "Male", 1, 0)
    z_test = np.where(test_raw[:, 9] == "Male", 1, 0)

    X_train = np.delete(train_raw, [9, 14], axis=1)
    X_test = np.delete(test_raw, [9, 14], axis=1)

    num_cols = [0, 2, 4, 9, 10, 11]

    for col in num_cols:
        X_train[:, col] = X_train[:, col].astype(float)
        X_test[:, col] = X_test[:, col].astype(float)

    for col in num_cols:
        mean = X_train[:, col].astype(float).mean()
        std = X_train[:, col].astype(float).std()
        X_train[:, col] = (X_train[:, col].astype(float) - mean) / std
        X_test[:, col] = (X_test[:, col].astype(float) - mean) / std

    def one_hot(train_col, test_col):
        values = np.unique(train_col)
        train_enc = np.zeros((train_col.shape[0], len(values)))
        test_enc = np.zeros((test_col.shape[0], len(values)))
        for i, v in enumerate(values):
            train_enc[:, i] = (train_col == v).astype(int)
            test_enc[:, i] = (test_col == v).astype(int)
        return train_enc, test_enc

    parts_train = []
    parts_test = []

    for i in range(X_train.shape[1]):
        if i in num_cols:
            parts_train.append(X_train[:, i].astype(float).reshape(-1,1))
            parts_test.append(X_test[:, i].astype(float).reshape(-1,1))
        else:
            tr_enc, te_enc = one_hot(X_train[:, i], X_test[:, i])
            parts_train.append(tr_enc)
            parts_test.append(te_enc)

    X_train_final = np.hstack(parts_train)
    X_test_final = np.hstack(parts_test)

    return X_train_final.astype(float), z_train.astype(float), y_train.astype(float), \
           X_test_final.astype(float), z_test.astype(float), y_test.astype(float)

def run(path):
    Xtr = np.load(os.path.join(path, "X_train.npy"))
    ztr = np.load(os.path.join(path, "z_train.npy"))
    ytr = np.load(os.path.join(path, "y_train.npy"))
    Xte = np.load(os.path.join(path, "X_test.npy"))
    zte = np.load(os.path.join(path, "z_test.npy"))
    yte = np.load(os.path.join(path, "y_test.npy"))

    Xtr = add_bias(Xtr)
    Xte = add_bias(Xte)

    print("Unconstrained model:")
    th0 = train_lr(Xtr, ytr)
    yhat0 = pred(Xte, th0)
    print("acc =", acc(yte, yhat0))
    print("p%-rule =", prule(yhat0, zte))
    print("----------------")

    cs = [0.8, 0.5, 0.2, 0]
    for c in cs:
        th = train_fair(Xtr, ytr, ztr, c)
        yhat = pred(Xte, th)
        print("c =", c)
        print("acc =", acc(yte, yhat))
        print("p%-rule =", prule(yhat, zte))
        print("----------------")



if __name__ == "__main__":
    folders = [
        "synth_phi_3.142",
        "synth_phi_1.571",
        "synth_phi_0.785",
        "synth_phi_0.524",
        "synth_phi_0.393"
    ]

    for f in folders:
        print("dataset:", f)
        run(f)

    print("dataset: adult")
    Xtr, ztr, ytr, Xte, zte, yte = load_adult_dataset()

    Xtr = add_bias(Xtr)
    Xte = add_bias(Xte)

    print("Unconstrained model:")
    th0 = train_lr(Xtr, ytr)
    yhat0 = pred(Xte, th0)
    print("acc =", acc(yte, yhat0))
    print("p%-rule =", prule(yhat0, zte))

    cs = [0.8, 0.5, 0.2, 0]
    for c in cs:
        th = train_fair(Xtr, ytr, ztr, c)
        yhat = pred(Xte, th)
        print("c =", c)
        print("acc =", acc(yte, yhat))
        print("p%-rule =", prule(yhat, zte))