# AI Romantic Relationship Simulator (Research Project)

An exploration into marital dynamics using Reinforcement Learning. This project simulates a marriage between two agents from ages 25 to 80, subjected to random life events. Rather than using hard-coded responses, agents utilize a Neural Network policy to optimize for long-term happiness and marriage stability.

## 🔬 Research Goal
To determine if an AI can "learn" optimal interpersonal behaviors (responses) based on fixed internal traits ($X$) and fluctuating environmental variables ($Y$).

## 🛠 Model Architecture

### 1. Variable Set X (Internal Traits)
Each agent is initialized with a unique set of non-linear variables:
* **Cognitive:** IQ, Rational Thinking.
* **Emotional:** EQ, Emotional Reasoning, Kindness, Ability to Love.
* **Moral/Stability:** Faithfulness, Responsibility, Mental Stability.
* **External Factor:** Kids.

### 2. Variable Set Y (Environmental/Outcome)
* **Status:** Wealth, Pressure.
* **Relational:** Love/Support.
* **Global Metrics:** Happiness, Marriage Stability.

### 3. The Learning Mechanism (πθ)
The simulation uses a **Policy Gradient** approach. 
* **State ($S$):** A vector containing $[X, Y, \text{Current Event}]$.
* **Action ($A$):** The agent's response to an event, selected by the neural network.
* **Reflection:** If a Delta $Y$ threshold is met, the system triggers a "Reflection" phase where internal weights (or $X$ variables) are adjusted based on the gradient of the reward.

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* PyTorch (for the Neural Network)
* Gymnasium (for the environment wrapper)
