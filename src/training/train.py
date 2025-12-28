import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

MODEL_PATH = "/models/linear.pt"

def make_data(n=2000, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 1)).astype(np.float32)
    # True relationship: y = 3x + 2 + noise
    y = (3.0 * x + 2.0 + rng.normal(0, noise, size=(n, 1))).astype(np.float32)
    return x, y

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("linear-regression-demo")

    lr = 0.05
    epochs = 50

    x_np, y_np = make_data()
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr=lr)

    with mlflow.start_run():
        mlflow.log_params({"lr": lr, "epochs": epochs})

        for epoch in range(epochs):
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            mlflow.log_metric("mse", float(loss.item()), step=epoch)

        os.makedirs("/models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

        w = float(model.weight.item())
        b = float(model.bias.item())
        mlflow.log_metrics({"learned_weight": w, "learned_bias": b})

        print(f"Saved model to {MODEL_PATH}")
        print(f"Learned: y ≈ {w:.3f}x + {b:.3f}")

if __name__ == "__main__":
    main()