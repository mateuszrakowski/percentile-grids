import os
import random
from datetime import datetime, timedelta

import pandas as pd


def create_mock():
    df = pd.read_csv("src/grids/data_mock/original_mock.csv")

    # Set new name
    first_names = ["John", "Jane", "Bob", "Alice", "Mark", "Emily"]
    last_names = ["Doe", "Smith", "Johnson", "Williams", "Brown", "Jones"]

    name = random.choice(first_names) + "_" + random.choice(last_names)

    df.rename(columns={"XXX": name}, inplace=True)

    study_date = df.loc[3, name]
    birth_date = df.loc[0, name]

    datetime_study_date = datetime.strptime(study_date, "%Y-%m-%d %H:%M")
    datetime_birth_date = datetime.strptime(birth_date, "%Y-%m-%d")

    study_range = random.randint(-1095, 1)
    birth_date = random.randint(-5475, 1825)

    new_study_date = datetime_study_date + timedelta(days=study_range)
    new_birth_date = datetime_birth_date + timedelta(days=birth_date)

    # Set new dates
    df.loc[3, name] = new_study_date.strftime("%Y-%m-%d %H:%M")
    df.loc[0, name] = new_birth_date.strftime("%Y-%m-%d")

    # Set new patient ID
    df.loc[1, name] = str(random.randint(100000, 999999))

    # Set new volume values with 5% maximum change
    for row in df[7:].itertuples():
        five_percent = float(df.loc[row.Index, "Unnamed: 2"]) * 0.15
        new_value = float(df.loc[row.Index, "Unnamed: 2"]) + random.uniform(
            -five_percent, five_percent
        )

        df.loc[row.Index, "Unnamed: 2"] = round(new_value, 3)

    # Save file
    if not os.path.exists("src/grids/data_mock/artifical_reference_dataset"):
        os.makedirs("src/grids/data_mock/artifical_reference_dataset")

    df.to_csv(
        f"src/grids/data_mock/artifical_reference_dataset/{name}_{df.loc[1, name]}.csv",
        index=False,
    )


if __name__ == "__main__":
    for i in range(200):
        create_mock()
