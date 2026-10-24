"""java.util.Random.nextGaussian stream used by the Android image pipelines."""
import math
import numpy as np


def java_noise(seed, count):
    state = (int(seed) ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def bits(n):
        nonlocal state
        state = (state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return state >> (48 - n)

    def uniform():
        return ((bits(26) << 27) + bits(27)) / (1 << 53)

    out = []
    while len(out) < count:
        x, y = 2 * uniform() - 1, 2 * uniform() - 1
        radius = x * x + y * y
        if radius >= 1 or radius == 0:
            continue
        scale = math.sqrt(-2 * math.log(radius) / radius)
        out.extend((x * scale, y * scale))
    return np.asarray(out[:count], np.float64).astype(np.float32)
