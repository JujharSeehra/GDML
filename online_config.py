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

FPS = 30

FRAME_STACK = 6
ACTION_LOOKAHEAD = 2

BASE_MODEL_PATH = "models/dash_strong_baseline.pth"

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
    "recordings/session_004",
]

VAL_SESSION = "recordings/session_005"


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

EXPLORATION_RATE = 0.01
UNCERTAINTY_THRESHOLD = 0.75


# Elite self-imitation
MIN_ELITE_SECONDS = 3.0

ELITE_FRACTION_OF_BEST = 0.93

MAX_ELITE_FILES = 40


# Online fine-tuning
RETRAIN_EVERY_ATTEMPTS = 25

ONLINE_EPOCHS = 2

BATCH_SIZE = 64

LEARNING_RATE = 0.00005


# Dataset importance
EXPERT_SAMPLE_WEIGHT = 1.0

CORRECTION_SAMPLE_WEIGHT = 3.0

ELITE_SAMPLE_WEIGHT = 1.0

MAX_HOLD_CLASS_WEIGHT = 3.0
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