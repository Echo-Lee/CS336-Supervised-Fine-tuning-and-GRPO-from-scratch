import math

def learning_rate_schedule(t, alpha_max, alpha_min, t_warm, t_cold):
    if t < t_warm:
        return t * alpha_max / t_warm
    elif t <= t_cold:
        return alpha_min + .5 * (1 + math.cos((t - t_warm)/(t_cold - t_warm) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min