import retro
import numpy as np
import tensorflow as tf
from gym import Env
from gym.spaces import Box, MultiBinary

class StreetFighter(Env): 
    def __init__(self):
        super().__init__()
        # Specify action space and observation space 
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        # Startup and instance of the game 
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
    
    def reset(self):
        # Return the first frame 
        obs = self.game.reset()
        obs = self.preprocess(obs) 
        self.previous_frame = obs 
        
        # Create an attribute to hold the score delta 
        self.score = 0 
        return obs
    
    # def preprocess(self, observation): 
    #     # Grayscaling and resizing using TensorFlow
    #     gray = tf.image.rgb_to_grayscale(observation)
    #     resize = tf.image.resize(gray, [84, 84])
    #     return resize.numpy()
    
    def preprocess(self, observation):
        """Preprocess the game observation."""
        # Convert to TensorFlow tensor
        observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        
        # Grayscale the image
        gray = tf.image.rgb_to_grayscale(observation_tensor)
        
        # Resize to 84x84
        resized = tf.image.resize(gray, [84, 84], method=tf.image.ResizeMethod.BILINEAR)
        
        # Add channel dimension (if required by the model)
        reshaped = tf.reshape(resized, [84, 84, 1])
        
        return reshaped.numpy()
    
    def step(self, action): 
        # Take a step 
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs) 
        
        # Frame delta using TensorFlow
        frame_delta = tf.subtract(obs, self.previous_frame).numpy()
        self.previous_frame = obs 
        
        # Reshape the reward function
        reward = info['score'] - self.score 
        self.score = info['score'] 
        
        return frame_delta, reward, done, info
    
    def render(self, *args, **kwargs):
        self.game.render()
        
    def close(self):
        self.game.close()