import os
from typing import List, Optional, Union
import pandas as pd


class DataLoaderError(Exception):
    """Raised when a dataset file cannot be loaded, is empty, or is invalid."""
    pass


class DataLoader:
    """
    Dataset loader supporting CSV (.csv) and Excel (.xlsx) formats.
    Provides deterministic sheet selection, encoding fallback, and quality checks.
    """

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

    @classmethod
    def load_file(cls, file_path: str, sheet_name: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Loads a tabular dataset from a file path (.csv or .xlsx).
        Detects file extension and delegates to the appropriate loader.
        """
        if not file_path or not os.path.exists(file_path):
            raise DataLoaderError(f"File not found or invalid path: '{file_path}'")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".csv":
            return cls.load_csv(file_path)
        elif ext == ".xlsx":
            return cls.load_excel(file_path, sheet_name=sheet_name)
        elif ext == ".xls":
            try:
                import xlrd  # noqa: F401
                return cls.load_excel(file_path, sheet_name=sheet_name)
            except ImportError:
                raise DataLoaderError(
                    "Legacy Excel format (.xls) requires 'xlrd'. Please convert your file to .xlsx or .csv."
                )
        else:
            raise DataLoaderError(
                f"Unsupported file format '{ext}'. VeriLens supports .csv and .xlsx files."
            )

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Loads a CSV dataset with encoding fallback and empty file detection.
        Maintains backward compatibility with all existing callers.
        """
        if not file_path or not os.path.exists(file_path):
            raise DataLoaderError(f"File not found: '{file_path}'")

        if os.path.getsize(file_path) == 0:
            raise DataLoaderError("The uploaded CSV file is empty (0 bytes).")

        encodings = ["utf-8", "latin1", "cp1252"]

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                if df.empty or len(df.columns) == 0:
                    raise DataLoaderError("The uploaded CSV file contains no data.")
                df.attrs["filename"] = os.path.basename(file_path)
                return df
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                raise DataLoaderError("The uploaded CSV file is empty.")

        raise DataLoaderError(
            "Unable to read CSV. Unsupported file encoding."
        )

    @classmethod
    def get_sheet_names(cls, file_path: str) -> List[str]:
        """Returns list of sheet names in an Excel workbook."""
        if not file_path or not os.path.exists(file_path):
            raise DataLoaderError(f"File not found: '{file_path}'")
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            names = list(wb.sheetnames)
            wb.close()
            return names
        except Exception as e:
            raise DataLoaderError(f"Corrupted or invalid Excel file: {e}")

    @classmethod
    def load_excel(cls, file_path: str, sheet_name: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Loads an Excel workbook (.xlsx).
        Inspects sheet names and selects the requested sheet or defaults
        to the first non-empty sheet.
        """
        if not file_path or not os.path.exists(file_path):
            raise DataLoaderError(f"File not found: '{file_path}'")

        if os.path.getsize(file_path) == 0:
            raise DataLoaderError("The uploaded Excel file is empty (0 bytes).")

        sheet_names = cls.get_sheet_names(file_path)
        if not sheet_names:
            raise DataLoaderError("Excel workbook contains no sheets.")

        chosen_sheet: Optional[Union[str, int]] = None

        if sheet_name is not None:
            if isinstance(sheet_name, str) and sheet_name not in sheet_names:
                raise DataLoaderError(
                    f"Sheet '{sheet_name}' not found. Available sheets: {', '.join(sheet_names)}"
                )
            if isinstance(sheet_name, int) and (sheet_name < 0 or sheet_name >= len(sheet_names)):
                raise DataLoaderError(
                    f"Sheet index {sheet_name} out of range (total sheets: {len(sheet_names)})."
                )
            chosen_sheet = sheet_name
        else:
            # Deterministically find the first non-empty sheet using lightweight preview
            for name in sheet_names:
                try:
                    preview_df = pd.read_excel(file_path, sheet_name=name, nrows=5, engine="openpyxl")
                    if not preview_df.empty and len(preview_df.columns) > 0:
                        chosen_sheet = name
                        break
                except Exception:
                    continue

            if chosen_sheet is None:
                raise DataLoaderError("Excel workbook has no usable or non-empty sheets.")

        try:
            df = pd.read_excel(file_path, sheet_name=chosen_sheet, engine="openpyxl")
            if df.empty or len(df.columns) == 0:
                raise DataLoaderError(f"Sheet '{chosen_sheet}' is empty.")
            df.attrs["filename"] = os.path.basename(file_path)
            df.attrs["sheet_name"] = str(chosen_sheet)
            return df
        except DataLoaderError:
            raise
        except Exception as e:
            raise DataLoaderError(f"Failed to read sheet '{chosen_sheet}': {e}")