
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


## Project Structure

```
.
├── PPO_StreetFighter.ipynb           # Notebook for PPO training
├── A2C_streetfighter.ipynb           # Notebook for A2C training
├── Best_model                        # Folder containing the best-trained models
│   ├── a2c_streetfighter_model.zip   # Best model for A2C
│   ├── best_model_4200000.zip        # Best PPO model at 4.2 million steps
│   ├── best_model_4210000.zip        # Best PPO model at 4.21 million steps
│   ├── best_model_4220000.zip        # Best PPO model at 4.22 million steps
│   ├── best_model_5000000.zip        # Best PPO model at 5 million steps
├── logs                              # Folder containing tensorboard logs
│   ├── tensorboard-A2C               # Logs for A2C training
│   ├── tensorboard-PPO               # Logs for PPO training
```

---
## How to Run

### PPO Training
To train the agent using PPO, open and run `PPO_StreetFighter.ipynb` in a Jupyter Notebook environment.

### A2C Training
To train the agent using A2C, open and run `A2C_streetfighter.ipynb` in a Jupyter Notebook environment.

## Best Models

- The best-trained models are stored in the `Best_model` folder.
- Each model can be reloaded and evaluated using the corresponding notebook.
- Example usage for loading a model:
    ```python
    from stable_baselines3 import PPO
    model = PPO.load('./Best_model/best_model_5000000.zip')
    ```

## Logs

- Tensorboard logs are stored in the `logs` folder.
- To visualize the training process:
    ```bash
    tensorboard --logdir=logs
    ```

## Notes

- **PPO** was found to outperform **A2C** in this environment due to its robustness and the clipping mechanism.
- **A2C** results were moderate, and further tuning or experimentation may improve its performance.

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
- Change reward to health bar instead of score.
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
