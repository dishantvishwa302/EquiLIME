import numpy as np
import os
import csv
import pickle
import random

# Roll Number: 25CS60R38

def load_raw_data(path):
    features = []
    labels = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            labels.append(1 if row['y'] == 'yes' else 0)
            features.append(row)
    return features, np.array(labels)

def preprocess(features, labels):
    z = np.array([1 if int(row['age']) >= 39 else 0 for row in features])
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    mapping = {col: sorted(list(set([row[col] for row in features]))) for col in cat_cols}
    stats = {}
    for col in num_cols:
        col_data = np.array([float(row[col]) for row in features])
        stats[col] = (np.mean(col_data), np.std(col_data))
        if stats[col][1] == 0: stats[col] = (stats[col][0], 1.0)
    
    def get_feature_vector(row):
        vec = []
        for col in num_cols:
            mean, std = stats[col]
            vec.append((float(row[col]) - mean) / std)
        for col in cat_cols:
            for v in mapping[col]:
                vec.append(1 if row[col] == v else 0)
        return np.array(vec)

    X = np.array([get_feature_vector(row) for row in features])
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X, labels, z, stats, mapping, get_feature_vector

def train_lr(X, y, lr=0.1, ep=2000): # Increased epochs
    n, d = X.shape
    th = np.zeros(d)
    y_scaled = np.where(y == 1, 1, -1)
    for i in range(ep):
        s = X @ th
        yz = y_scaled * s
        ez = np.exp(np.clip(yz, -500, 500))
        m = 1 / (1 + ez)
        g = (- X.T @ (y_scaled * m)) / n
        th = th - lr * g
    return th

def train_fair(X, y, z, gamma, lr=0.1, ep=2000): # Increased epochs
    n, d = X.shape
    th = np.zeros(d)
    y_scaled = np.where(y == 1, 1, -1)
    zc = z - np.mean(z)
    v = (X.T @ zc) / n
    for i in range(ep):
        s = X @ th
        yz = y_scaled * s
        ez = np.exp(np.clip(yz, -500, 500))
        m = 1 / (1 + ez)
        g = (- X.T @ (y_scaled * m)) / n
        val = np.dot(th, v)
        penalty_grad = gamma * np.sign(val) * v
        th = th - lr * (g + penalty_grad)
    return th

def prule(yhat, z):
    y1 = yhat[z == 1]
    y0 = yhat[z == 0]
    p1 = np.mean(y1 == 1) if len(y1) > 0 else 0
    p0 = np.mean(y0 == 1) if len(y0) > 0 else 0
    if p0 == 0 or p1 == 0: return 0
    return min(p1/p0, p0/p1) * 100

def pred_prob(X, th):
    s = X @ th
    return 1 / (1 + np.exp(-np.clip(s, -500, 500)))

def pred(X, th):
    return np.where(X @ th >= 0, 1, 0)

def get_interpretable(row):
    return np.array([
        1 if int(row['age']) >= 39 else 0,
        1 if int(row['balance']) >= 448 else 0,
        1 if row['housing'] == 'yes' else 0,
        1 if row['education'] in ['primary', 'secondary'] else 0,
        1 if row['marital'] == 'married' else 0,
        1 if row['loan'] == 'yes' else 0
    ])

def perturb(x_prime, n=100):
    neighborhood = [x_prime]
    for _ in range(n-1):
        z_prime = x_prime.copy()
        idx = random.randint(0, len(x_prime) - 1)
        z_prime[idx] = 1 - z_prime[idx]
        neighborhood.append(z_prime)
    return np.array(neighborhood)

def reconstruct(z_prime, row_orig):
    row = row_orig.copy()
    if z_prime[0] == 1:
        if int(row_orig['age']) < 39: row['age'] = '45'
    else:
        if int(row_orig['age']) >= 39: row['age'] = '30'
    if z_prime[1] == 1:
        if int(row_orig['balance']) < 448: row['balance'] = '1000'
    else:
        if int(row_orig['balance']) >= 448: row['balance'] = '0'
    row['housing'] = 'yes' if z_prime[2] == 1 else 'no'
    if z_prime[3] == 1:
        if row_orig['education'] not in ['primary', 'secondary']: row['education'] = 'secondary'
    else:
        if row_orig['education'] in ['primary', 'secondary']: row['education'] = 'tertiary'
    row['marital'] = 'married' if z_prime[4] == 1 else 'single'
    row['loan'] = 'yes' if z_prime[5] == 1 else 'no'
    return row

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    data_path = "/Users/jarvis/IIT KGP Sem 2/AI and Ethics/AIETH Assignment/bank+marketing/bank-full.csv"
    features, labels = load_raw_data(data_path)
    X, y, z, stats, mapping, get_vec = preprocess(features, labels)
    
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_size = int(0.8 * n)
    tr_idx, te_idx = indices[:train_size], indices[train_size:]
    
    Xtr, ytr, ztr = X[tr_idx], y[tr_idx], z[tr_idx]
    Xte, yte, zte = X[te_idx], y[te_idx], z[te_idx]
    
    print("Training Accurate...")
    th_acc = train_lr(Xtr, ytr)
    print("Training Fair...")
    th_fair = train_fair(Xtr, ytr, ztr, gamma=5.0) # Increased gamma to ensure disagreement
    
    yhat_acc = pred(Xte, th_acc)
    yhat_fair = pred(Xte, th_fair)
    yhat_tr_acc = pred(Xtr, th_acc)
    yhat_tr_fair = pred(Xtr, th_fair)

    print(f"Acc-Model  - Train Acc: {np.mean(yhat_tr_acc == ytr):.4f}, Test Acc: {np.mean(yhat_acc == yte):.4f}, p%-rule: {prule(yhat_acc, zte):.2f}")
    print(f"Fair-Model - Train Acc: {np.mean(yhat_tr_fair == ytr):.4f}, Test Acc: {np.mean(yhat_fair == yte):.4f}, p%-rule: {prule(yhat_fair, zte):.2f}")
    
    disagree = np.where(yhat_acc != yhat_fair)[0]
    agree = np.where(yhat_acc == yhat_fair)[0]
    
    print(f"Number of disagreements: {len(disagree)}")
    
    sel_te_indices = []
    if len(disagree) >= 2:
        sel_te_indices.extend(disagree[:2].tolist())
    else:
        sel_te_indices.extend(disagree.tolist())
        
    if len(agree) >= 1:
        sel_te_indices.append(agree[0].tolist())
    
    # Fill to 5
    i = len(sel_te_indices)
    while len(sel_te_indices) < 5:
        if i < len(disagree):
            sel_te_indices.append(disagree[i].tolist())
        elif (i - len(disagree)) < len(agree):
            sel_te_indices.append(agree[i - len(disagree) + 1].tolist())
        else: break
        i += 1
    
    sel_te_indices = [int(i) for i in sel_te_indices]
    print(f"Selected test set indices: {sel_te_indices}")
    
    task_y_results = []
    for idx_te in sel_te_indices:
        raw_row = features[te_idx[idx_te]]
        x_prime = get_interpretable(raw_row)
        neighborhood_z_prime = perturb(x_prime, 100)
        
        probs_acc = []
        probs_fair = []
        for z_p in neighborhood_z_prime:
            row_recon = reconstruct(z_p, raw_row)
            x_recon = get_vec(row_recon)
            x_recon = np.insert(x_recon, 0, 1.0)
            
            probs_acc.append(pred_prob(x_recon, th_acc))
            probs_fair.append(pred_prob(x_recon, th_fair))
            
        task_y_results.append({
            'idx': idx_te,
            'x_prime': x_prime,
            'neighborhood': neighborhood_z_prime,
            'probs_acc': np.array(probs_acc),
            'probs_fair': np.array(probs_fair)
        })
        
    with open("task_xy_data.pkl", "wb") as f:
        pickle.dump({
            'results': task_y_results,
            'th_acc': th_acc,
            'th_fair': th_fair,
            'stats': stats,
            'mapping': mapping,
            'all_features': features,
            'te_idx': te_idx,
            'X': X,
            'y': y,
            'z': z
        }, f)
    print("Done Task XY.")
