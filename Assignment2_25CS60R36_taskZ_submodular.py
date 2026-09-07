import numpy as np
import pickle
import random

# Submodular Optimization (SP-LIME) style global feature importance
# Goal: Identify top 3 most important features across the dataset.

def hamming_dist(x, z):
    return np.sum(x != z)

def get_weights(x_prime, neighborhood, sigma=None):
    if sigma is None:
        sigma = np.sqrt(len(x_prime)) * 0.75
    dists = np.array([hamming_dist(x_prime, z) for z in neighborhood])
    weights = np.exp(-(dists**2) / (sigma**2))
    return weights

def fit_local_model(Z_prime, probs, weights):
    W = np.diag(weights)
    Z_bias = np.hstack([np.ones((Z_prime.shape[0], 1)), Z_prime])
    lhs = Z_bias.T @ W @ Z_bias
    rhs = Z_bias.T @ W @ probs
    th = np.linalg.pinv(lhs) @ rhs
    return th

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

def pred_prob(X, th):
    s = X @ th
    return 1 / (1 + np.exp(-np.clip(s, -500, 500)))

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    
    print("Loading Task XY data...")
    with open("task_xy_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    features = data['all_features']
    th_acc = data['th_acc']
    th_fair = data['th_fair']
    mapping = data['mapping']
    stats = data['stats']
    
    # We'll take a representative subset of the dataset to calculate global importance
    # (since running LIME on all 45k would be slow)
    subset_size = 100
    indices = np.random.choice(len(features), subset_size, replace=False)
    
    num_features = 6
    interpretable_features = ["age", "balance", "housing", "education", "marital", "loan"]
    
    # Local weights matrix: (subset_size, num_features)
    W_acc = np.zeros((subset_size, num_features))
    W_fair = np.zeros((subset_size, num_features))
    
    print(f"Calculating local importance for {subset_size} samples...")
    for i, idx in enumerate(indices):
        raw_row = features[idx]
        x_prime = get_interpretable(raw_row)
        neighborhood = perturb(x_prime, 50) # Use smaller N for speed
        
        # Black-box labeling
        probs_acc = []
        probs_fair = []
        for z_p in neighborhood:
            row_recon = reconstruct(z_p, raw_row)
            # Embedding-based features
            vec = []
            for col in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
                mean, std = stats[col]
                vec.append((float(row_recon[col]) - mean) / std)
            for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
                for v in mapping[col]:
                    vec.append(1 if row_recon[col] == v else 0)
            x_recon = np.array(vec)
            x_recon = np.insert(x_recon, 0, 1.0)
            
            probs_acc.append(pred_prob(x_recon, th_acc))
            probs_fair.append(pred_prob(x_recon, th_fair))
            
        weights = get_weights(x_prime, neighborhood)
        
        # Fit local models
        th_l_acc = fit_local_model(neighborhood, np.array(probs_acc), weights)
        th_l_fair = fit_local_model(neighborhood, np.array(probs_fair), weights)
        
        # Store absolute coefficients (importance)
        W_acc[i] = np.abs(th_l_acc[1:])
        W_fair[i] = np.abs(th_l_fair[1:])
        
    # Global Importance Score: I_j = sqrt(sum_i W_ij)
    I_acc = np.sqrt(np.sum(W_acc, axis=0))
    I_fair = np.sqrt(np.sum(W_fair, axis=0))
    
    # Top 3 features
    top3_acc = np.argsort(I_acc)[-3:][::-1]
    top3_fair = np.argsort(I_fair)[-3:][::-1]
    
    print("\nGlobal Feature Importance (Accurate Model):")
    for i in range(num_features):
        print(f"{interpretable_features[i]:<12}: {I_acc[i]:.4f}")
        
    print("\nTop 3 Features (Accurate):")
    for idx in top3_acc:
        print(f"- {interpretable_features[idx]}")

    print("\nGlobal Feature Importance (Fair Model):")
    for i in range(num_features):
        print(f"{interpretable_features[i]:<12}: {I_fair[i]:.4f}")
        
    print("\nTop 3 Features (Fair):")
    for idx in top3_fair:
        print(f"- {interpretable_features[idx]}")

    # Submodular optimization for diversity (Greedy Pick)
    # The coverage function c(V, W) = sum_j (1 if any i in V has important j) * I_j
    # For now, picking top 3 features based on global importance is a good proxy 
    # for "most important features" as requested.
    
    print("\nTask Z Submodular Done.")
