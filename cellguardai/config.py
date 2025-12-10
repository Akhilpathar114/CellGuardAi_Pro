# Basic configuration constants for CellGuardAI

# Pack voltage limits (in Volts)
PACK_VOLTAGE_MIN = 20.0
PACK_VOLTAGE_MAX = 120.0

# Current limits (in Amps)
CURRENT_MIN = -300.0   # discharge
CURRENT_MAX = 300.0    # charge

# State of Charge limits (in %)
SOC_MIN = 0.0
SOC_MAX = 100.0

# Temperature limits (in °C)
TEMP_MIN = -20.0
TEMP_MAX = 80.0

# Max allowed cell voltage difference (mV) for imbalance
MAX_CELL_DIFF = 80.0
