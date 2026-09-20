from .mrl_lookup import find_mrl
from .risk_engine import assess_mrl
from .decay_engine import predict_residue # Import our new math module

def assess_crop_safety(crop, pesticide, initial_residue, spray_date, target_date=None):
    """
    Complete AgriShield MRL assessment with time-based decay.
    """
    # 1. Find the applicable MRL and half-life
    mrl_data = find_mrl(crop, pesticide)

    if mrl_data is None:
        return {
            "status": "UNKNOWN",
            "message": "No MRL data found for this crop and pesticide."
        }

    # 2. Calculate decayed residue
    try:
        predicted_residue = predict_residue(
            initial_residue=initial_residue,
            half_life_days=mrl_data["half_life_days"],
            spray_date=spray_date,
            target_date=target_date
        )
    except ValueError as e:
        return {"status": "ERROR", "message": str(e)}

    # 3. Compare predicted residue with MRL
    result = assess_mrl(
        residue=predicted_residue,
        mrl=mrl_data["mrl_mg_per_kg"]
    )

    return {
        "crop": mrl_data["crop"],
        "pesticide": mrl_data["pesticide"],
        "spray_date": spray_date,
        "target_date": target_date if target_date else "Today",
        "initial_residue_mg_per_kg": initial_residue,
        "predicted_residue_mg_per_kg": predicted_residue,
        "mrl_mg_per_kg": mrl_data["mrl_mg_per_kg"],
        "percentage_of_mrl": result["percentage_of_mrl"],
        "status": result["status"],
        "message": result["message"],
        "source": mrl_data["source"]
    }

if __name__ == "__main__":

    result = assess_crop_safety(
        crop="Tomato",
        pesticide="Lambda cyhalothrin",
        predicted_residue=0.075
    )

    print(result)
