

## 1. Project Overview

**Topic:** Comparative Analysis of RL Algorithms in the Snake Game  

**Algorithms:**  
- Tabular Q-Learning  
- PPO  
- Policy Gradient

**Environment:**  
- `gym-snakegame` (Gymnasium)

**Team Structure:**  
- 3 Agent Developers  
- 1 Lead Analyst / Reporter

---

## 2. File Organization

```text
/snake-rl-project
│
├── /agents
│   ├── q_learning.ipynb
│   ├── ppo.ipynb
│   └── policy_gradient.ipynb
│
├── /final_results
│   ├── /models
│   ├── /data
│   └── /plots
│
├── main_comparison.ipynb
├── rewards.py
├── README.md
└── requirements.txt
```

## 3. Standardized Reward Systems

To ensure **scientific validity**, all three algorithms will be tested across **three distinct reward functions**.

### System A: Sparse (Baseline)

Designed to test the agent's ability to handle **delayed gratification**.

| Event | Reward |
|------|------|
| Food Consumption | +10.0 |
| Collision (Death) | -10.0 |
| Time Step | 0 |

---

### System B: Dense (Shaped)

Designed for **faster convergence** through constant feedback.

| Event | Reward |
|------|------|
| Food Consumption | +10.0 |
| Collision (Death) | -10.0 |
| Proximity Reward | +-0.1 (based on distance to food using Euclidean distance) |
| Step Penalty | -0.01 (discourages looping / idling) |

---

### System C: Survivalist

Designed to prioritize **safety and longevity** over speed.

| Event | Reward |
|------|------|
| Food Consumption | +5.0 |
| Collision (Death) | -20.0 |
| Survival Reward | +0.05 per step survived |

---

## 4. Evaluation Metrics

The **best trained version** of each agent will be evaluated across **100 test episodes**.

| Metric | Scientific Purpose |
|------|------|
| Average Return | Measures overall performance stability |
| Max Snake Length | Measures peak performance capability |
| Steps per Food | Measures pathfinding efficiency *(lower is better)* |
| Success Rate (%) | Ratio of episodes reaching snake length > 10 |
| Generalization Gap | Performance drop when moving from **10×10** grid to **20×20** grid |

---

- **Hyperparameter Logging:**  
Each notebook must explicitly list:
- Learning Rate 
- Discount Factor 
- Batch Size
