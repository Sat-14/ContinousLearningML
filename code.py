# ==============================================================
# Continual Learning on CIFAR-100 using Pretrained ViT
# Part 1: Model & Training (Naive, EWC, Replay)
# Author: Satwik Rai & [Teammate Name]
# ==============================================================

import torch
import torch.nn as nn
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.strategies import Naive, EWC
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger
from timm import create_model
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 1. SETUP
# --------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Split CIFAR-100 into 5 tasks of 20 classes each
benchmark = SplitCIFAR100(n_experiences=5, return_task_id=False, seed=42)

interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

# --------------------------------------------------------------
# 2. MODEL DEFINITION (Pretrained Vision Transformer)
# --------------------------------------------------------------
vit = create_model('vit_base_patch16_224', pretrained=True)
vit.head = nn.Linear(vit.head.in_features, 100)
vit.to(device)

criterion = nn.CrossEntropyLoss()

# --------------------------------------------------------------
# 3. STRATEGY CONFIGURATION
# --------------------------------------------------------------

def get_strategy(strategy_name, model, lr=0.001, ewc_lambda=0.4, mem_size=2000):
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    
    if strategy_name == "naive":
        return Naive(model, optimizer, criterion, train_mb_size=32, train_epochs=1,
                     evaluator=eval_plugin, device=device)
    
    elif strategy_name == "ewc":
        return EWC(model, optimizer, criterion, ewc_lambda=ewc_lambda,
                   train_mb_size=32, train_epochs=1, evaluator=eval_plugin, device=device)
    
    elif strategy_name == "replay":
        replay_plugin = ReplayPlugin(mem_size=mem_size)
        return Naive(model, optimizer, criterion, train_mb_size=32, train_epochs=1,
                     evaluator=eval_plugin, plugins=[replay_plugin], device=device)
    else:
        raise ValueError("Unknown strategy name")

# --------------------------------------------------------------
# 4. TRAINING LOOP
# --------------------------------------------------------------

strategies = {
    "naive": get_strategy("naive", vit),
    "ewc": get_strategy("ewc", vit, ewc_lambda=1.0),
    "replay": get_strategy("replay", vit, mem_size=1000)
}

results = []

for name, strategy in strategies.items():
    print(f"\n===== Training Strategy: {name.upper()} =====")
    for experience in benchmark.train_stream:
        print(f"\n--- Task {experience.current_experience+1} ---")
        strategy.train(experience)
        eval_results = strategy.eval(benchmark.test_stream)
        acc = eval_results['Top1_Acc_Stream/eval_phase/test_stream']
        results.append({
            "Strategy": name,
            "Task": experience.current_experience+1,
            "Accuracy": acc
        })
        print(f"Task {experience.current_experience+1} Accuracy: {acc:.2f}%")

# --------------------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------------------

df = pd.DataFrame(results)
df.to_csv("results_model.csv", index=False)
print("\nResults saved to results_model.csv")

# --------------------------------------------------------------
# 6. VISUALIZE BASIC RESULTS
# --------------------------------------------------------------
plt.figure(figsize=(8,5))
for strategy_name in df['Strategy'].unique():
    sub_df = df[df['Strategy'] == strategy_name]
    plt.plot(sub_df['Task'], sub_df['Accuracy'], marker='o', label=strategy_name.upper())

plt.title("Continual Learning Accuracy (CIFAR-100)")
plt.xlabel("Task Number")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()
