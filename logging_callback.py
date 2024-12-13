import os
import json
from stable_baselines3.common.callbacks import BaseCallback

class LoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self):
        # Check if an episode ended
        if self.locals.get("done", False):
            episode_reward = self.locals.get("rewards", 0)
            episode_length = self.locals.get("info", {}).get("l", 0)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
        return True

    def save_logs(self):
        # Save episodic metrics
        logs = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths
        }
        with open(os.path.join(self.log_dir, "episodic_logs.json"), "w") as f:
            json.dump(logs, f, indent=4)
