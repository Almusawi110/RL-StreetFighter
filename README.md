
# Street Fighter Reinforcement Learning Agent

This project demonstrates the use of reinforcement learning algorithms to train an agent to play **Street Fighter** using OpenAI Gym's retro environments. The project primarily explores the **Proximal Policy Optimization (PPO)** and **Advantage Actor-Critic (A2C)** algorithms, comparing their performances and analyzing the challenges faced during the training process.

---

## Features
- **Custom Environment**: A tailored Gym environment for Street Fighter using `retro`.
- **Reinforcement Learning Algorithms**: Implementation of PPO and A2C algorithms with tuned hyperparameters.
- **Model Training**: Training models over millions of timesteps to optimize the agent's performance.
- **Performance Evaluation**: Comparing algorithms based on rewards, gameplay performance, and stability.
- **Logging and Monitoring**: TensorBoard integration for tracking metrics during training.
- **Challenges and Insights**: Documenting the process, challenges, and solutions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Almusawi110/RL-StreetFighter.git
   cd RL-StreetFighter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the **Street Fighter** environment:
   - Download the ROM for Street Fighter and place it in the correct directory.
   - Ensure `retro` is configured to detect the ROM.

---

## Training the Agent

### PPO Training
To train the agent using PPO:
```bash
python train_ppo.py
```

### A2C Training
To train the agent using A2C:
```bash
python train_a2c.py
```

---

## File Structure
```
.
├── street_fighter_env.py    # Custom environment setup for Street Fighter
├── train_ppo.py             # Training script for PPO
├── train_a2c.py             # Training script for A2C
├── Best_model/                  # Saved models
├── logs/                    # Training logs and TensorBoard files
├── requirements.txt         # Required Python libraries
└── README.md                # Project documentation
```

---

## Hyperparameters
The project uses fine-tuned hyperparameters based on extensive experimentation. Below are the key hyperparameters used:

| Algorithm | n_steps | Gamma | Learning Rate |
|-----------|---------|-------|---------------|
| PPO       | 2819    | 0.859 | 5e-07         |
| A2C       | 2819    | 0.859 | 5e-07         |

---

## Evaluation and Results

### PPO
- Achieved high average rewards (~34,700).
- Demonstrated robust gameplay performance, clearing multiple levels.
- Outperformed A2C in both stability and learning efficiency.

### A2C
- Moderate performance with average rewards (~29,400).
- Faced challenges in learning and gameplay, requiring further tuning.

---

## Challenges
1. Setting up `retro` for an older game like Street Fighter.
2. Training without GPUs, leading to long training times on CPUs.
3. Adjusting learning rates and hyperparameters to stabilize performance.

---

## Future Work
- Experiment with additional algorithms like DDPG or SAC.
- Fine-tune A2C further to enhance its performance.
- Optimize the training pipeline for GPU acceleration.
- Explore curriculum learning to guide the agent through levels progressively.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- [OpenAI Gym](https://github.com/openai/gym) for providing the retro environment framework.
- The Street Fighter ROM and retro community for enabling AI experiments on classic games.
