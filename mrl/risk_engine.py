def assess_mrl(residue, mrl):
    """
    Assess pesticide residue against the applicable MRL.

    residue: predicted/measured residue in mg/kg
    mrl: permitted MRL in mg/kg
    """

    if mrl <= 0:
        raise ValueError("MRL must be greater than zero.")

    if residue < 0:
        raise ValueError("Residue cannot be negative.")

    percentage = (residue / mrl) * 100

    if percentage < 70:
        status = "SAFE"
        message = "Residue is comfortably below the MRL."

    elif percentage < 90:
        status = "WARNING"
        message = "Residue is approaching the MRL."

    elif percentage <= 100:
        status = "NEAR_LIMIT"
        message = "Residue is very close to the MRL."

    else:
        status = "DANGER"
        message = "Predicted residue exceeds the MRL."

    return {
        "residue": residue,
        "mrl": mrl,
        "percentage_of_mrl": round(percentage, 2),
        "status": status,
        "message": message
    }


# Test example
if __name__ == "__main__":

    result = assess_mrl(
        residue=0.19,
        mrl=0.20
    )

    print(result)
