import pandas as pd


class DataLoader:

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:

        encodings = ["utf-8", "latin1", "cp1252"]

        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)

            except UnicodeDecodeError:
                continue

        raise Exception(
            "Unable to read CSV. Unsupported file encoding."
        )