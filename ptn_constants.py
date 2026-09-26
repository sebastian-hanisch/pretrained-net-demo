"""Konstanten der Demo "Vortrainiertes Netz" (Stück 11 der Zeitreihen-Prognose-Linie): Vehikel "Tagesaufträge mehrerer Depots", Netz, Vortraining, Feintuning, Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095
FIRST_TEST = 730
LEVEL = 100.0
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
SHIFTED_PATTERN = (0.60, 0.60, 0.65, 0.70, 0.90, 1.60, 1.55)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)
HOLIDAY_DROP = 0.5
HOLIDAY_REBOUND = 0.15
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3
SEASON_PERIOD = 7
AR_PHI = 0.98

# --- Glättung aus Stück 2 (Vergleich) ------------------------------------------------------------------------------------------------------------
ETS_MODELS = {"hw_mult": ("add", "mul")}
INIT_DAYS = 28
INIT_WEEKS = 8
FIT_STAGE1 = 3000
FIT_TOP = 6
FIT_ROUNDS = 6
FIT_PER_START = 40
FIT_SEED = 20240924
PHI_MIN, PHI_MAX = 0.80, 0.98

# --- Netz ------------------------------------------------------------------------------------------------------------------------------------------
HORIZON = 14                  # Prognosehorizont in Tagen
WINDOW = 56                   # Rückblick-Fenster in Tagen (acht Wochen)
GBM_CONTEXT = 91              # das Boosting-Modell aus Stück 6 braucht 91 Tage Vorlauf
VAL_DAYS = 45                 # die letzten Tage vor dem Testjahr, nur für die Lernkurve des Vortrainings
COVARIATES = ("window", "full")
COVARIATE_NAMES = {"window": "nur Rückblick-Fenster (und Wochentag)", "full": "Fenster + Kalender und Aktionsplan"}

# --- Regler und Voreinstellungen ------------------------------------------------------------------------------------------------------------------
POOL_CHOICES, DEFAULT_POOL = (3, 10, 20, 40, 70, 100), 40
NEW_MIN, NEW_MAX, NEW_STEP, DEFAULT_NEW = 5, 30, 5, 10
HIST_CHOICES, DEFAULT_HIST = (98, 140, 182, 273, 365, 548, 730), 182
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.04, 0.4, 0.02, 0.14
SHIFT_MIN, SHIFT_MAX, SHIFT_STEP, DEFAULT_SHIFT = 0.0, 1.0, 0.25, 0.0
EPOCHS_MIN, EPOCHS_MAX, EPOCHS_STEP, DEFAULT_EPOCHS = 4, 40, 4, 12
BLOCKS_MIN, BLOCKS_MAX, DEFAULT_BLOCKS = 1, 4, 2
WIDTH_CHOICES = (16, 32, 64, 128)
DEFAULT_WIDTH = 32
FT_MIN, FT_MAX, FT_STEP, DEFAULT_FT = 0, 1000, 50, 100
LR_PRETRAIN = 2e-3
LR_FINETUNE = 1e-4

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------------
EXP_SEEDS = tuple(range(3))
EXP_NEW = 10
HISTORIES = (98, 182, 365, 730)
POOL_SIZES = (3, 10, 30, 100)
SHIFT_LEVELS = (0.0, 0.5, 1.0)
SHIFT_FT = 300
FT_LEVELS = (0, 50, 100, 300, 1000)
FT_CONDITIONS = ((182, 0.0), (730, 0.0), (182, 1.0), (730, 1.0))
