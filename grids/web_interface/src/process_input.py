import pandas as pd


def process_input(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    head = (
        df.head(5)
        .drop(columns=["Unnamed: 2"])
        .dropna()
        .set_index("Pacjent")
        .T.reset_index(drop=True)
        .iloc[:, [1, 0, 2, 3]]
    )

    body = df[7:].copy()
    body.columns = df.iloc[6].tolist()
    body = body.set_index("Struktura").T.iloc[1:].reset_index(drop=True)

    head.rename(
        columns={
            "Identyfikator pacjenta": "PatientID",
            "Data urodzenia": "BirthDate",
            "Data badania": "StudyDate",
            "Opis badania": "StudyDescription",
        },
        inplace=True,
    )

    body.columns = [
        f"{col.replace(" – ", "_").replace(" - ", "_").replace(" ", "_").replace("-", '_')}"
        for col in body.columns
    ]

    body = pd.concat([head, body], axis=1)
    return head, body
