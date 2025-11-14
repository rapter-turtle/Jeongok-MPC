import numpy as np



def sample_int_scalar(mean, var):
    rng = np.random.default_rng()
    x = rng.normal(loc=mean, scale=np.sqrt(var))
    return int(round(x))

print(sample_int_scalar(1, 1))  # mean=1, variance=3
