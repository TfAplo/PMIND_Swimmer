import gymnasium as gym
import numpy as np
from collections import deque

class FeatureFilterWrapper(gym.ObservationWrapper):
    def __init__(self, env, idx):
        super().__init__(env)
        self.idx = idx

        old_space = env.observation_space
        self.keep = [i for i in range(old_space.shape[0]) if i != idx]

        self.observation_space = gym.spaces.Box(
            low=old_space.low[self.keep],
            high=old_space.high[self.keep],
            dtype=old_space.dtype
        )

    def observation(self, observation):
        return observation[self.keep]

class ObsTimeExtensionWrapper(gym.ObservationWrapper):
    def __init__(self, env, size):
        super().__init__(env)

        self.size = size
        self.buffer = deque(maxlen=size + 1)

        old_space = env.observation_space

        self.obs_dim = old_space.shape[0]
        
        low = np.repeat(old_space.low, size + 1)
        high = np.repeat(old_space.high, size + 1)
    
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=old_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.buffer.clear()
        for i in range(self.size):
            null_obs = np.zeros(self.obs_dim, dtype=obs.dtype)
            self.buffer.append(null_obs)
        self.buffer.append(obs)

        return self._get_observation(), info

    def observation(self, observation):
        self.buffer.append(observation)
        return self._get_observation()

    def _get_observation(self):
        return np.concatenate(list(self.buffer), axis=0)

class ActionTimeExtensionWrapper(gym.Wrapper):
    def __init__(self, env, M):
        super().__init__(env)

        self.M = M
        old_space = env.action_space

        self.action_dim = old_space.shape

        low = np.tile(old_space.low, M)
        high = np.tile(old_space.high, M)

        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=old_space.dtype
        )

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        # reshape en (M, action_dim)
        if len(self.action_dim) == 1:
            action = action.reshape(self.M, self.action_dim[0])
        else:
            action = action.reshape((self.M,) + self.action_dim)

        first_action = action[0]

        return self.env.step(first_action)
    
class VelocityControlWrapper(gym.Wrapper):
    def __init__(self, env, velocity_multiplier=10.0):
        super().__init__(env)
        self.velocity_multiplier = velocity_multiplier

    def step(self, action):
        # On vérifie qu'on accède bien aux données MuJoCo
        if hasattr(self.unwrapped, 'data'):
             # On multiplie l'action [-1, 1] par le gain pour obtenir une vraie vitesse
            target_velocity = action * self.velocity_multiplier
            
            # L'action devient directement la vitesse (sur les axes X et Y)
            self.unwrapped.data.qvel[:len(action)] = target_velocity
            
            # On envoie une action "vide" (zéro force) au step classique 
            # pour que le moteur physique calcule juste les collisions et la nouvelle position
            null_action = np.zeros_like(action)
            return self.env.step(null_action)
        else:
            # si l'env n'est pas basé sur mujoco
            return self.env.step(action)