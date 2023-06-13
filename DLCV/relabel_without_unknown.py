import numpy as np

unknown_indices = np.array([4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78])

known_indices = np.delete(np.arange(80), unknown_indices)

resort_indices = np.zeros(80)
for i in np.arange(80):
    if i in known_indices:
        resort_indices[i] = np.where(known_indices==i)[0]
    else:
        resort_indices[i] = None