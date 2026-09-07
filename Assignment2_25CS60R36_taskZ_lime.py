import numpy as np
import pickle

# LIME implementation from scratch

def hamming_dist(x, z):
    return np.sum(x != z)

def get_weights(x_prime, neighborhood, sigma=None):
    if sigma is None:
        sigma = np.sqrt(len(x_prime)) * 0.75
    dists = np.array([hamming_dist(x_prime, z) for z in neighborhood])
    weights = np.exp(-(dists**2) / (sigma**2))
    return weights

def fit_local_model(Z_prime, probs, weights):
    # Z_prime: (N, K) interpretable features
    # probs: (N,) black-box target (probabilities)
    # weights: (N,) sample weights
    # Fit weighted linear regression: th = (Z'T W Z')^-1 Z'T W y
    W = np.diag(weights)
    # Add intercept to Z_prime
    Z_bias = np.hstack([np.ones((Z_prime.shape[0], 1)), Z_prime])
    
    # Equation: (X.T W X) theta = X.T W y
    lhs = Z_bias.T @ W @ Z_bias
    rhs = Z_bias.T @ W @ probs
    
    # Solve lhs * th = rhs
    # Use pseudo-inverse for stability
    th = np.linalg.pinv(lhs) @ rhs
    return th

if __name__ == "__main__":
    with open("task_xy_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    results = data['results']
    interpretable_features = ["age", "balance", "housing", "education", "marital", "loan"]
    
    all_explanations = []
    
    print("Fitting local models for each instance...")
    for res in results:
        idx = res['idx']
        x_prime = res['x_prime']
        neighborhood = res['neighborhood']
        probs_acc = res['probs_acc']
        probs_fair = res['probs_fair']
        
        weights = get_weights(x_prime, neighborhood)
        
        th_acc = fit_local_model(neighborhood, probs_acc, weights)
        th_fair = fit_local_model(neighborhood, probs_fair, weights)
        
        all_explanations.append({
            'idx': idx,
            'th_acc': th_acc,
            'th_fair': th_fair
        })
        
        print(f"\nInstance {idx} Explanations:")
        print(f"{'Feature':<12} | {'Acc-Model':>10} | {'Fair-Model':>10}")
        print("-" * 40)
        for i, name in enumerate(interpretable_features):
            # i+1 because th[0] is intercept
            print(f"{name:<12} | {th_acc[i+1]:>10.4f} | {th_fair[i+1]:>10.4f}")

    # Save for submodular optimization
    with open("task_z_lime_data.pkl", "wb") as f:
        pickle.dump({
            'all_explanations': all_explanations,
            'interpretable_features': interpretable_features
        }, f)
    print("\nTask Z LIME Part 1 Done.")
