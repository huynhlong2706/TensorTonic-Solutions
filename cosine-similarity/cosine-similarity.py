import numpy as np


def cosine_similarity(a: list, b: list) -> float:
  """Returns the cosine similarity as a Python float."""
  a_arr = np.asarray(a, dtype=float)
  b_arr = np.asarray(b, dtype=float)

  # Calculate the dot product
  dot_prod = float(np.dot(a_arr, b_arr))

  # Calculate the product of the magnitudes (norms)
  norm_a = np.linalg.norm(a_arr)
  norm_b = np.linalg.norm(b_arr)

  # Avoid division by zero if an input vector is all zeros
  if norm_a == 0 or norm_b == 0:
    return 0.0

  return float(dot_prod / (norm_a * norm_b))
