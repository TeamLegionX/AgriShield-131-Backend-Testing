import csv
from pathlib import Path


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "mrl_data.csv"


def find_mrl(crop, pesticide):
    crop = crop.strip().lower()
    pesticide = pesticide.strip().lower()

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row["crop"].strip().lower() == crop
                and row["pesticide"].strip().lower() == pesticide
            ):
                return {
                    "crop": row["crop"],
                    "pesticide": row["pesticide"],
                    "mrl_mg_per_kg": float(row["mrl_mg_per_kg"]),
                    "half_life_days": float(row["half_life_days"]),  # <-- NEW
                    "source": row["source"]
                }

    return None


if __name__ == "__main__":

    result = find_mrl(
        "Tomato",
        "Lambda cyhalothrin"
    )

    print(result)
