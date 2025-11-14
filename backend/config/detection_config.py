"""
Detection configuration constants.
These thresholds can be tuned for better performance.
"""
# Per-face probability threshold above which a face is considered suspicious
FACE_THRESHOLD = 0.5

# Extremely suspicious threshold (high confidence detection)
HIGH_THRESHOLD = 0.9

# Minimum number of faces needed for a confident decision
MIN_FACES_FOR_DECISION = 2

# Number of suspicious faces needed to consider video manipulated
SUSPICIOUS_COUNT_NEEDED = 2

# 90th percentile threshold for verdict decision
P90_THRESHOLD = 0.7

