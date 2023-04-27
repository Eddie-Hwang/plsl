import math


class LinearNoiseScheduler:
    def __init__(self, initial_noise_std, final_noise_std, total_steps, **kwargs):
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.total_steps = total_steps + 1

    def step(self, current_step):
        progress = (current_step + 1) / self.total_steps
        current_noise_std = self.initial_noise_std - progress * (self.initial_noise_std - self.final_noise_std)
        return current_noise_std
    

class ExponentialNoiseScheduler:
    def __init__(self, initial_noise_std, final_noise_std, total_steps, sharpness_factor=1., **kwargs):
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.total_steps = total_steps + 1
        self.sharpness_factor = sharpness_factor
        # self.decay_rate = (final_noise_std / initial_noise_std) ** (1.0 / (total_steps * sharpness_factor))
        self.decay_rate = (final_noise_std / initial_noise_std) ** (1.0 / total_steps)

    def step(self, current_step):
        current_noise_std = self.initial_noise_std * (self.decay_rate ** (current_step * self.sharpness_factor))
        return current_noise_std


class CosineAnnealingNoiseScheduler:
    def __init__(self, initial_noise_std, final_noise_std, total_steps, sharpness_factor=1., **kwargs):
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.total_steps = total_steps + 1
        self.sharpness_factor = sharpness_factor

    def step(self, current_step):
        progress = (current_step + 1) / self.total_steps
        cosine_term = (1 + math.cos(self.sharpness_factor * math.pi * progress)) / 2
        current_noise_std = self.final_noise_std + (self.initial_noise_std - self.final_noise_std) * cosine_term
        return current_noise_std
    

class ConstantNoiseScheduler:
    def __init__(self, initial_noise_std, **kwargs):
        self.noise_std = initial_noise_std

    def step(self, **kwargs):
        return self.noise_std