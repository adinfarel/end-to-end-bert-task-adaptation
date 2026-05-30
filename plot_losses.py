'''
./plot_losses.py

Plotting loss curve into plot graph to make understandable information >.<.
'''

import matplotlib.pyplot as plt
import os

os.makedirs('docs/assets', exist_ok=True)

# Pre-training loss
pretrain_epochs     = list(range(1, 16))
pretrain_train_loss = [8.4425, 8.0304, 7.4008, 6.9573, 6.6442,
                       7.0352, 5.9668, 6.5900, 5.0658, 6.1593,
                       5.7733, 5.7631, 6.3017, 6.1311, 6.6235]

eval_steps      = [3, 6, 9, 12, 15]
eval_loss_vals  = [7.1409, 6.7867, 6.4233, 6.1101, 6.0180]

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(pretrain_epochs, pretrain_train_loss,
        color="#4C9BE8", linewidth=2.5, marker='o', markersize=5,
        label="Train loss (per epoch)")

ax.plot(eval_steps, eval_loss_vals,
        color="#E87B4C", linewidth=2.5, marker='o', markersize=7,
        label="Eval loss (every 3 epoch)")

ax.fill_between(pretrain_epochs, pretrain_train_loss, alpha=0.08, color="#4C9BE8")
ax.set_title("AlmondBERT — Pre-Training Loss (MLM)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_xlim(0.5, 15.15)
ax.set_ylim(4.0, 9.5)
ax.set_xticks(pretrain_epochs)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)

# Annotate
baseline = 9.21 # ln(vocab_size)
ax.axhline(y=baseline, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
ax.text(1.2, baseline + 0.1, f"Random baseline = {baseline}", fontsize=9, color='gray')

ax.annotate(f"Final eval: {eval_loss_vals[-1]:.4f}",
            xy=(eval_steps[-1], eval_loss_vals[-1]),
            xytext=(-80, 20), textcoords="offset points",
            fontsize=10, color='#E87B4C',
            arrowprops=dict(arrowstyle="->", color='#E87B4C', lw=1.2))

plt.tight_layout()
plt.savefig("docs/assets/pretrain_loss.png", dpi=150)
plt.close()
print("Saved: docs/assets/pretrain_loss.png")

# NER
ner_epochs     = [1, 2, 3, 4, 5]
ner_train_loss = [1.1564, 0.7692, 0.8399, 0.7242, 0.8374]
ner_eval_loss  = [0.8536, 0.7125, 0.6652, 0.6408, 0.6363]
 
fig, ax = plt.subplots(figsize=(9, 5))
 
ax.plot(ner_epochs, ner_train_loss,
        color="#57B894", linewidth=2.5, marker="o", markersize=6,
        label="Train Loss")
 
ax.plot(ner_epochs, ner_eval_loss,
        color="#9B59B6", linewidth=2.5, marker="s", markersize=6,
        label="Eval Loss")
 
ax.fill_between(ner_epochs, ner_train_loss, alpha=0.10, color="#57B894")
ax.fill_between(ner_epochs, ner_eval_loss,  alpha=0.10, color="#9B59B6")
 
ax.set_title("AlmondBERT — NER Fine-Tuning Loss", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_ylim(0.4, 1.4)
ax.set_xticks(ner_epochs)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
 
# Annotate finals
ax.annotate(f"Final train: {ner_train_loss[-1]:.4f}",
            xy=(ner_epochs[-1], ner_train_loss[-1]),
            xytext=(-90, 15), textcoords="offset points",
            fontsize=10, color="#57B894",
            arrowprops=dict(arrowstyle="->", color="#57B894", lw=1.2))
 
ax.annotate(f"Final eval: {ner_eval_loss[-1]:.4f}",
            xy=(ner_epochs[-1], ner_eval_loss[-1]),
            xytext=(-90, -25), textcoords="offset points",
            fontsize=10, color="#9B59B6",
            arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=1.2))
 
plt.tight_layout()
plt.savefig("docs/assets/ner_loss.png", dpi=150)
plt.close()
print("Saved: docs/assets/ner_loss.png")