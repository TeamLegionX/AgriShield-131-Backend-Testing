from datetime import datetime, date


def predict_residue(initial_residue: float, half_life_days: float, spray_date: str, target_date: str = None) -> float:
    """
    Calculates remaining pesticide residue using exponential decay.
    Dates must be in 'YYYY-MM-DD' format. If target_date is omitted, uses today.
    """
    if target_date is None:
        target_date = date.today().strftime('%Y-%m-%d')

    d_spray = datetime.strptime(spray_date, "%Y-%m-%d")
    d_target = datetime.strptime(target_date, "%Y-%m-%d")

    days_elapsed = (d_target - d_spray).days

    if days_elapsed < 0:
        raise ValueError("Target date cannot be before the spray date.")

    if half_life_days <= 0:
        raise ValueError("Half-life must be greater than zero.")

    # Apply the exponential decay formula
    remaining_residue = initial_residue * (0.5 ** (days_elapsed / half_life_days))

    return round(remaining_residue, 4)


# Quick test
if __name__ == "__main__":
    # Example: 2.0 mg/kg initial spray, 14-day half life, 21 days later
    result = predict_residue(2.0, 14.0, "2026-08-24", "2026-09-14")
    print(f"Predicted residue: {result} mg/kg")  # Should be ~0.7071