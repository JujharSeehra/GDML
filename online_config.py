# ---------------------------------------------------------
# SCREEN CAPTURE
# ---------------------------------------------------------
#
# IMPORTANT:
# Replace these values with the EXACT values from recorder.py.
#

CAPTURE_LEFT = 90
CAPTURE_TOP = 100
CAPTURE_WIDTH = 1000
CAPTURE_HEIGHT = 700


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

FRAME_STACK = 4
FPS = 30

BASE_MODEL_PATH = "models/dash_bc.pth"
ONLINE_MODEL_PATH = "models/dash_online.pth"
CANDIDATE_MODEL_PATH = "models/dash_online_candidate.pth"


# ---------------------------------------------------------
# ORIGINAL HUMAN DATA
# ---------------------------------------------------------

TRAIN_SESSIONS = [
    "recordings/session_001",
    "recordings/session_002",
    "recordings/session_003",
    "recordings/session_004",
    "recordings/session_005",
    "recordings/session_006",
]

VAL_SESSION = "recordings/session_007"


# ---------------------------------------------------------
# PROGRESS
# ---------------------------------------------------------
#
# If you know how many seconds a complete run takes,
# enter it here.
#
# Example:
#
# LEVEL_DURATION_SECONDS = 85.0
#
# If you don't know yet, leave None.
# The system will still use survival time as its score.
#

LEVEL_DURATION_SECONDS = None


# ---------------------------------------------------------
# DEATH DETECTION
# ---------------------------------------------------------

DEATH_MOTION_THRESHOLD = 2.0

DEATH_STATIC_FRAMES = 8

MIN_EPISODE_SECONDS = 2.0

RESTART_WAIT_SECONDS = 0.8


# ---------------------------------------------------------
# HUMAN CORRECTIONS
# ---------------------------------------------------------
#
# J = force RELEASE
# K = force HOLD
#
# These keys should not affect Geometry Dash itself.
#

CORRECTION_RELEASE_KEY = "j"
CORRECTION_HOLD_KEY = "k"

CORRECTION_SAVE_EVERY = 2


# ---------------------------------------------------------
# AUTONOMOUS EXPLORATION
# ---------------------------------------------------------
#
# Only occasionally explore when the model itself
# is uncertain.
#
# FOR YOUR FIRST TEST:
# change this to 0.0.
#
# Once death/restart works correctly:
# use around 0.03.
#

EXPLORATION_RATE = 0.00

UNCERTAINTY_THRESHOLD = 0.70


# ---------------------------------------------------------
# SELF-IMITATION
# ---------------------------------------------------------

MIN_ELITE_SECONDS = 3.0

# Save runs that reach at least 85% of the current best.
ELITE_FRACTION_OF_BEST = 0.85

MAX_ELITE_FILES = 25


# ---------------------------------------------------------
# ONLINE TRAINING
# ---------------------------------------------------------

RETRAIN_EVERY_ATTEMPTS = 9999

ONLINE_EPOCHS = 4

BATCH_SIZE = 64

LEARNING_RATE = 0.0003


# Original human demonstration
EXPERT_SAMPLE_WEIGHT = 1.0

# Your DAgger-style corrections
CORRECTION_SAMPLE_WEIGHT = 3.0

# Agent's successful self-generated examples
ELITE_SAMPLE_WEIGHT = 0.25


# Don't let HOLD class weighting explode.
MAX_HOLD_CLASS_WEIGHT = 6.0

# ---------------------------------------------------------
# ATTEMPT / EPISODE DETECTION
# ---------------------------------------------------------

ATTEMPT_BRIGHT_THRESHOLD = 210

ATTEMPT_MIN_BRIGHT_PIXELS = 15

ATTEMPT_PRESENT_FRAMES = 2

ATTEMPT_REARM_FRAMES = 5


# Don't include the final death animation frames
# in self-imitation training.
EPISODE_TAIL_TRIM_FRAMES = 24