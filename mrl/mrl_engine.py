import math

def calculate_residue(c0, dt50, days):
    k = math.log(2) / dt50
    residue = c0 * math.exp(-k * days)
    return residue

def calculate_safe_harvest_time(c0, dt50, mrl):
    if c0 <= mrl:
        return 0
    k = math.log(2) / dt50
    safe_days = math.log(c0 / mrl) / k
    return safe_days
