import numpy as np

arr = np.array([10, 3, 20, 4, 15,7,8])

# Pairwise absolute differences
distances = np.abs(arr[:, None] - arr[None, :])
print(distances)
# Ignore self-distances
distances = distances.astype(float)

np.fill_diagonal(distances, np.inf)
print(distances)
print(np.argmin(distances))
print("min : ",np.min(distances))
i, j = np.unravel_index(np.argmin(distances), distances.shape)
print(i,j)
print(arr[i], arr[j])
print("Distance:", distances[i, j])

#import numpy as np

# Sample points (rows are points)
X = np.array([
    [1, 2],
    [3, 4],
    [1.1, 2.1],
    [10, 10]
])

# Compute pairwise Euclidean distances
print(X[:, np.newaxis, :])
print(X[np.newaxis, :, :])
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
print(diff)
distances = np.linalg.norm(diff, axis=2)
print("distances : ",distances)
# Ignore self-distances
np.fill_diagonal(distances, np.inf)

# Find indices of the closest pair
i, j = np.unravel_index(np.argmin(distances), distances.shape)

print(f"Closest points: {i} and {j}")
print(f"Distance: {distances[i, j]}")
print("Point 1:", X[i])
print("Point 2:", X[j])