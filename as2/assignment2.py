import numpy as np


# Q1.1: Sigmoid Function
def sigmoid(x):
    """
    Implements the sigmoid (logistic) function.

    Args:
        x: A numpy array of shape/ (d,) or (N, d) representing the input data.

    Returns:
        y: A numpy array of the same shape as x, with sigmoid applied element-wise.
    """
    y = 1.0 / (1.0 + np.exp(-x))
    return y


# Q1.2: One Layer Inference (Logistic Regression)
def inference_layer(X, W, b):
    """
    Implements forward propagation for one layer: h = sigmoid(W @ X + b)

    Args:
        X: Input data of shape (d,) for a single sample or (d, N) for N samples.
        W: Weight matrix of shape (l, d).
        b: Bias vector of shape (l,).

    Returns:
        y: Output of shape (l,) for a single sample or (l, N) for N samples.
    """
    # W @ X gives shape (l,) or (l, N)
    # b needs to be reshaped for broadcasting when X is (d, N)
    z = W @ X + b.reshape(-1, 1) if X.ndim == 2 else W @ X + b
    y = sigmoid(z)
    return y


# Q1.3: Two Layer Inference (Neural Network)
def inference_2layers(X, W1, W2, b1, b2):
    """
    Implements forward propagation of a two-layer neural network.

    Args:
        X: Input data of shape (d,) or (d, N).
        W1: Weights for the first layer, shape (H, d).
        W2: Weights for the second layer, shape (1, H).
        b1: Bias for the first layer, shape (H,).
        b2: Bias for the second layer (scalar or shape (1,)).

    Returns:
        y: Output prediction(s), shape (1,) or (N,) — squeezed from (1, N).
    """
    # First layer: h = sigmoid(W1 @ X + b1), shape (H,) or (H, N)
    h = inference_layer(X, W1, b1)
    # Second layer: y = sigmoid(W2 @ h + b2), shape (1,) or (1, N)
    y = inference_layer(h, W2, np.atleast_1d(b2))
    # Squeeze to remove the leading dimension of 1
    y = np.squeeze(y, axis=0)
    return y


# Q1.4: Binary Cross-Entropy (BCE) Loss
def bce_forward(yhat, y):
    """
    Computes the binary cross-entropy loss.

    Args:
        yhat: Predicted outputs, numpy array of shape (N,).
        y:    Target labels, numpy array of shape (N,).

    Returns:
        loss_value: Scalar BCE loss.
    """
    N = y.shape[0]
    # Clip predictions to avoid log(0)
    eps = 1e-12
    yhat = np.clip(yhat, eps, 1 - eps)
    loss_value = -(1.0 / N) * np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    return loss_value


# Q3: Implement Gradients
def gradients(X, y, W1, W2, b1, b2):
    """
    Calculate the gradients of the BCE loss with respect to W1, W2, b1, b2.

    Args:
        X:  Input data of shape (d, N).
        y:  Target labels of shape (N,).
        W1: Weight matrix for the first layer, shape (H, d).
        W2: Weight matrix for the second layer, shape (1, H).
        b1: Bias for the first layer, shape (H,).
        b2: Bias for the second layer (scalar).

    Returns:
        dW1: Gradient of loss w.r.t. W1, shape (H, d).
        dW2: Gradient of loss w.r.t. W2, shape (1, H).
        db1: Gradient of loss w.r.t. b1, shape (H,).
        db2: Gradient of loss w.r.t. b2 (scalar).
        L:   BCE loss value (scalar).
    """
    N = X.shape[1]

    # Forward pass
    # Layer 1: h = sigmoid(W1 @ X + b1), shape (H, N)
    z1 = W1 @ X + b1.reshape(-1, 1)
    h = sigmoid(z1)

    # Layer 2: yhat = sigmoid(W2 @ h + b2), shape (1, N) -> squeeze to (N,)
    z2 = W2 @ h + b2
    yhat = sigmoid(z2).squeeze()          # shape (N,)

    # Compute loss
    L = bce_forward(yhat, y)

    # Backward pass
    # dL/dz2 = yhat - y, shape (N,)
    dz2 = yhat - y                        # shape (N,)

    # Gradient for W2: dL/dW2 = (1/N) * dz2 @ h^T, shape (1, H)
    dW2 = (1.0 / N) * dz2.reshape(1, -1) @ h.T

    # Gradient for b2: dL/db2 = (1/N) * sum(dz2), scalar
    db2 = (1.0 / N) * np.sum(dz2)

    # Backpropagate to hidden layer
    # delta1 = W2^T @ dz2 * h * (1 - h), shape (H, N)
    delta1 = (W2.T @ dz2.reshape(1, -1)) * h * (1 - h)

    # Gradient for W1: dL/dW1 = (1/N) * delta1 @ X^T, shape (H, d)
    dW1 = (1.0 / N) * delta1 @ X.T

    # Gradient for b1: dL/db1 = (1/N) * sum(delta1, axis=1), shape (H,)
    db1 = (1.0 / N) * np.sum(delta1, axis=1)

    return dW1, dW2, db1, db2, L


# Q4: Optimization — Gradient Descent Parameter Update
def update_params(batchx, batchy, W1, b1, W2, b2, lr=0.01):
    """
    Performs one step of gradient descent to update all parameters.

    Args:
        batchx: Mini-batch of features, shape (d, N).
        batchy: Corresponding mini-batch of labels, shape (N,).
        W1: Current W1, shape (H, d).
        b1: Current b1, shape (H,).
        W2: Current W2, shape (1, H).
        b2: Current b2 (scalar).
        lr: Learning rate, default 0.01.

    Returns:
        W1, b1, W2, b2: Updated parameters.
    """
    # Compute gradients
    dW1, dW2, db1, db2, L = gradients(batchx, batchy, W1, W2, b1, b2)

    # Gradient descent update
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    return W1, b1, W2, b2


# Q5: Putting it All Together — Training and Evaluation
def train_nn(filename, hidden_layer_size, iters=10000, lr=0.01, batch_size=64, seed=42):
    """
    Reads in Twitter data, extracts features, trains a two-layer neural network,
    and returns the learned parameters along with training/test loss curves.

    Args:
        filename:          Path to twitter_data.pkl
        hidden_layer_size: Number of hidden units (H)
        iters:             Number of training iterations
        lr:                Learning rate
        batch_size:        Mini-batch size
        seed:              Random seed for reproducibility

    Returns:
        W1, b1, W2, b2:   Trained parameters
    """
    import pickle
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt

    # Ensure NLTK data is available (needed by utils.py)
    import nltk
    import os
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    nltk.data.path.insert(0, nltk_data_dir)
    # Only download if not already present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

    # Need utils.py in the same directory or on path
    from as2_file.utils import extract_features

    np.random.seed(seed)

    # Load data
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    train_x = data['train_x']
    train_y = data['train_y'].flatten()   # shape (N_train,)
    test_x  = data['test_x']
    test_y  = data['test_y'].flatten()    # shape (N_test,)
    freqs   = data['freqs']

    # Extract features
    # extract_features returns (1, 3) per tweet: [bias, pos_count, neg_count]
    train_feats = np.array([extract_features(t, freqs).flatten() for t in train_x])  # (N_train, 3)
    test_feats  = np.array([extract_features(t, freqs).flatten() for t in test_x])   # (N_test, 3)

    # Transpose to (d, N) format as required by our network
    # d = 3 (bias, positive count, negative count)
    X_train = train_feats.T   # (3, N_train)
    X_test  = test_feats.T    # (3, N_test)

    print(f"Training samples: {X_train.shape[1]}, Test samples: {X_test.shape[1]}")
    print(f"Feature dimension: {X_train.shape[0]}")
    print(f"Hidden layer size: {hidden_layer_size}")
    print(f"Learning rate: {lr}, Iterations: {iters}, Batch size: {batch_size}")

    # Initialize parameters
    d = X_train.shape[0]  # 3
    H = hidden_layer_size
    W1 = np.random.randn(H, d) * 0.1
    b1 = np.zeros(H)
    W2 = np.random.randn(1, H) * 0.1
    b2 = 0.0

    # Training loop
    train_losses = []
    test_losses  = []
    N_train = X_train.shape[1]

    for i in range(int(iters)):
        # Mini-batch sampling
        idx = np.random.choice(N_train, batch_size, replace=True)
        batchx = X_train[:, idx]
        batchy = train_y[idx]

        # Update parameters
        W1, b1, W2, b2 = update_params(batchx, batchy, W1, b1, W2, b2, lr=lr)

        # Log losses every 100 iterations
        if i % 100 == 0:
            # Training loss (on full training set)
            yhat_train = inference_2layers(X_train, W1, W2, b1, b2)
            t_loss = bce_forward(yhat_train, train_y)
            train_losses.append(t_loss)

            # Test loss
            yhat_test = inference_2layers(X_test, W1, W2, b1, b2)
            e_loss = bce_forward(yhat_test, test_y)
            test_losses.append(e_loss)

            if i % 1000 == 0:
                # Compute training accuracy
                train_acc = np.mean((yhat_train > 0.5).astype(float) == train_y)
                test_acc  = np.mean((yhat_test > 0.5).astype(float) == test_y)
                print(f"Iter {i:6d} | Train Loss: {t_loss:.4f} | Test Loss: {e_loss:.4f} "
                      f"| Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Final evaluation
    yhat_train = inference_2layers(X_train, W1, W2, b1, b2)
    yhat_test  = inference_2layers(X_test, W1, W2, b1, b2)
    train_acc = np.mean((yhat_train > 0.5).astype(float) == train_y)
    test_acc  = np.mean((yhat_test > 0.5).astype(float) == test_y)
    print(f"\n=== Final Results ===")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")

    # Plot loss curves
    iters_axis = np.arange(len(train_losses)) * 100
    plt.figure(figsize=(10, 5))
    plt.plot(iters_axis, train_losses, label='Training Loss')
    plt.plot(iters_axis, test_losses, label='Test Loss')
    plt.xlabel('Iteration')
    plt.ylabel('BCE Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150)
    print("Loss curves saved to loss_curves.png")

    # Test examples
    print("\n=== Test Examples ===")
    example_tweets = [
        'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!',
        'I am happy because I am learning :)',
        'I am so sad and angry today',
        'This movie was absolutely wonderful and I loved every minute of it',
        'The food was terrible, worst restaurant experience ever',
    ]
    for tweet in example_tweets:
        feat = extract_features(tweet, freqs)  # (1, 3)
        x_vec = feat.flatten()                 # (3,)
        y_hat = inference_2layers(x_vec, W1, W2, b1, b2)
        sentiment = 'Positive sentiment' if y_hat > 0.5 else 'Negative sentiment'
        print(f"  Tweet: \"{tweet}\"")
        print(f"  Prediction: {y_hat:.4f} -> {sentiment}\n")

    # Save model parameters
    params = {
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2,
    }
    with open('as2_output/assignment2.pkl', 'wb') as f:
        pickle.dump(params, f)
    print("Model parameters saved to assignment2.pkl")

    return W1, b1, W2, b2
