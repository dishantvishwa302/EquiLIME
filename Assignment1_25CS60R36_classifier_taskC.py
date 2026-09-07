import numpy as np
import os

def add_bias(X):
    n = X.shape[0]
    ones = np.ones((n, 1))
    return np.hstack((ones, X))

def grad_log(X, y, th):
    n = X.shape[0]
    s = X @ th
    yz = y * s
    ez = np.exp(np.clip(yz, -500, 500))
    den = 1 + ez
    m = 1 / den
    g = - X.T @ (y * m)
    return g / n

def train_unconstrained(X, y, lr=0.01, ep=1000):
    d = X.shape[1]
    th = np.zeros(d)
    for _ in range(ep):
        g = grad_log(X, y, th)
        th = th - lr * g
    return th

def train_penalty(X, y, z, gamma, lr=0.01, ep=1000):
    n, d = X.shape
    th = np.zeros(d)
    zc = z - np.mean(z)
    v = (X.T @ zc) / n
    for _ in range(ep):
        g = grad_log(X, y, th)
        val = np.dot(th, v)
        if val > 0:
            sign = 1
        elif val < 0:
            sign = -1
        else:
            sign = 0
        penalty_grad = gamma * sign * v
        g_total = g + penalty_grad
        th = th - lr * g_total
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
    if p1 == 0 or p0 == 0:
        return 0
    r1 = p1 / p0
    r2 = p0 / p1
    return min(r1, r2) * 100

def run_synth(path):
    Xtr = np.load(os.path.join(path, "X_train.npy"))
    ztr = np.load(os.path.join(path, "z_train.npy"))
    ytr = np.load(os.path.join(path, "y_train.npy"))
    Xte = np.load(os.path.join(path, "X_test.npy"))
    zte = np.load(os.path.join(path, "z_test.npy"))
    yte = np.load(os.path.join(path, "y_test.npy"))
    Xtr = add_bias(Xtr)
    Xte = add_bias(Xte)
    print("Unconstrained model:")
    th0 = train_unconstrained(Xtr, ytr)
    yhat_tr = pred(Xtr, th0)
    yhat_te = pred(Xte, th0)
    train_acc = acc(ytr, yhat_tr)
    test_acc = acc(yte, yhat_te)
    print("train acc =", train_acc)
    print("test acc =", test_acc)
    print("p%-rule =", prule(yhat_te, zte))
    print("----------------")
    gammas = [0, 0.5, 1, 1.5]
    for g in gammas:
        th = train_penalty(Xtr, ytr, ztr, g)
        yhat_tr = pred(Xtr, th)
        yhat_te = pred(Xte, th)
        train_acc = acc(ytr, yhat_tr)
        test_acc = acc(yte, yhat_te)
        print("gamma =", g)
        print("train acc =", train_acc)
        print("test acc =", test_acc)
        print("p%-rule =", prule(yhat_te, zte))
        print("----------------")
def load_adult_dataset():
    def read_file(path):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 15:
                    continue
                rows.append(parts)
        return np.array(rows)
    train_raw = read_file(os.path.join("adult", "adult.data"))
    test_raw = read_file(os.path.join("adult", "adult.test"))
    if test_raw[0][0].startswith("|"):
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
    for c in num_cols:
        X_train[:, c] = X_train[:, c].astype(float)
        X_test[:, c] = X_test[:, c].astype(float)
        mean = X_train[:, c].astype(float).mean()
        std = X_train[:, c].astype(float).std()
        X_train[:, c] = (X_train[:, c].astype(float) - mean) / std
        X_test[:, c] = (X_test[:, c].astype(float) - mean) / std

    def one_hot(col_tr, col_te):
        vals = np.unique(col_tr)
        tr = np.zeros((col_tr.shape[0], len(vals)))
        te = np.zeros((col_te.shape[0], len(vals)))
        for i, v in enumerate(vals):
            tr[:, i] = (col_tr == v).astype(int)
            te[:, i] = (col_te == v).astype(int)
        return tr, te
    parts_tr = []
    parts_te = []

    for i in range(X_train.shape[1]):
        if i in num_cols:
            parts_tr.append(X_train[:, i].astype(float).reshape(-1, 1))
            parts_te.append(X_test[:, i].astype(float).reshape(-1, 1))
        else:
            enc_tr, enc_te = one_hot(X_train[:, i], X_test[:, i])
            parts_tr.append(enc_tr)
            parts_te.append(enc_te)
    X_train_final = np.hstack(parts_tr)
    X_test_final = np.hstack(parts_te)
    return (
        X_train_final.astype(float),
        z_train.astype(float),
        y_train.astype(float),
        X_test_final.astype(float),
        z_test.astype(float),
        y_test.astype(float)
    )

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
        run_synth(f)
    print("dataset: adult")
    print("dataset: adult")
    Xtr, ztr, ytr, Xte, zte, yte = load_adult_dataset()
    Xtr = add_bias(Xtr)
    Xte = add_bias(Xte)
    print("Unconstrained model:")
    th0 = train_unconstrained(Xtr, ytr)
    yhat_tr = pred(Xtr, th0)
    yhat_te = pred(Xte, th0)
    print("train acc =", acc(ytr, yhat_tr))
    print("test acc =", acc(yte, yhat_te))
    print("p%-rule =", prule(yhat_te, zte))
    print("----------------")
    gammas = [0, 0.5, 1, 1.5]
    for g in gammas:
        th = train_penalty(Xtr, ytr, ztr, g)
        yhat_tr = pred(Xtr, th)
        yhat_te = pred(Xte, th)
        print("gamma =", g)
        print("train acc =", acc(ytr, yhat_tr))
        print("test acc =", acc(yte, yhat_te))
        print("p%-rule =", prule(yhat_te, zte))
        print("----------------")