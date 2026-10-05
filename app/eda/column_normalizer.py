import re
from typing import List, Set, Optional, Any
import pandas as pd


# Common business/data abbreviations and their canonical semantic stems
_SEMANTIC_ALIASES = {
    "cust": "customer",
    "customerid": "customer_id",
    "prod": "product",
    "productid": "product_id",
    "num": "number",
    "no": "number",
    "nbr": "number",
    "qty": "quantity",
    "amt": "amount",
    "pct": "percent",
    "cat": "category",
    "dept": "department",
    "dt": "date",
    "addr": "address",
    "desc": "description",
    "rev": "revenue",
    "emp": "employee",
    "org": "organization",
    "ord": "order",
    "orderid": "order_id",
    "itm": "item",
    "itemid": "item_id",
    "id": "id",
    "idx": "index",
    "cnt": "count",
    "val": "value",
    "yr": "year",
    "mo": "month",
    "cd": "code",
}

# Regex patterns matching technical row/dataframe index columns
_INDEX_NAME_PATTERNS = [
    r"^unnamed(?::|\s|_|\b|\d|$)",    # Unnamed: 0, unnamed:0, unnamed_0, unnamed
    r"^(index|idx)$",                  # index, idx, Index, INDEX
    r"^row[_\s]?(num(ber)?|id|no)$",   # row_number, row_num, rownum, row_id, rowid, row_no, rowno
    r"^record[_\s]?(num(ber)?|id|no)$",# record_number, record_num, record_id, recordid, record_no
    r"^line[_\s]?(num(ber)?|no)$",     # line_number, line_num, lineno
    r"^level_\d+$",                    # level_0, level_1
]
_INDEX_NAME_REGEX = re.compile("|".join(f"(?:{p})" for p in _INDEX_NAME_PATTERNS), re.IGNORECASE)

# Business entity prefixes that indicate legitimate business identifiers
_BUSINESS_ENTITY_PREFIXES = {
    "customer", "cust", "client", "user", "account", "member", "subscriber", "consumer",
    "order", "ord", "invoice", "inv", "bill", "receipt", "transaction", "txn", "trans", "payment",
    "product", "prod", "item", "article", "part", "material", "merchandise", "device", "asset",
    "employee", "emp", "staff", "agent", "vendor", "supplier", "partner", "merchant", "seller",
    "store", "shop", "branch", "warehouse", "depot", "facility", "location", "site", "station",
    "shipment", "delivery", "tracking", "package", "parcel", "flight", "trip", "vehicle",
    "patient", "claim", "policy", "contract", "deal", "ticket", "session", "visit", "event",
    "country", "state", "city", "postal", "zip",
}

# Standalone codes that are domain business identifiers
_STANDALONE_BUSINESS_KEYS = {
    "sku", "asin", "upc", "ean", "isbn", "vin", "uuid", "guid",
}


class ColumnNormalizer:
    """
    Deterministic column-name normalizer for schema alignment and entity matching.
    Transforms column names into canonical semantic representations.
    """

    @classmethod
    def is_business_identifier(cls, col_name: str) -> bool:
        """
        Determines whether a column name represents a legitimate business identifier
        (e.g., customer_id, order_id, product_id, sku, asin, invoice_id, transaction_id).
        Never returns True for technical index columns (e.g. index, row_id, record_id, unnamed: 0).
        """
        if not col_name:
            return False

        col_str = str(col_name).strip()
        raw_lower = col_str.lower()

        # Never classify technical index/row columns as business identifiers
        if _INDEX_NAME_REGEX.search(raw_lower) or raw_lower.startswith("unnamed"):
            return False

        # Standalone domain keys or standard entity ID
        if raw_lower in _STANDALONE_BUSINESS_KEYS or raw_lower in ("id", "code", "key"):
            return True

        norm = cls.normalize(col_str)
        if norm in _STANDALONE_BUSINESS_KEYS or norm in ("id", "code", "key"):
            return True

        tokens = cls.tokenize(col_str)
        if not tokens:
            return False

        # Any column ending with _id, _code, _key, _number whose prefix is not an index term
        if norm.endswith("_id") or norm.endswith("_code") or norm.endswith("_key") or norm.endswith("_number"):
            prefix = norm.rsplit("_", 1)[0]
            if prefix not in ("row", "record", "line", "index", "level", "unnamed"):
                return True

        # Check if any token represents a known business entity
        has_entity = any(t in _BUSINESS_ENTITY_PREFIXES for t in tokens)
        has_id_indicator = any(t in ("id", "number", "num", "no", "code", "key") for t in tokens)

        if has_entity and has_id_indicator:
            return True

        return False

    @classmethod
    def is_index_like(cls, col_name: str, series: Optional[Any] = None) -> bool:
        """
        Determines whether a column is an index-like column (row counter, dataframe index,
        unnamed export column, or sequential numeric index) rather than a business identifier.
        """
        if col_name is None:
            return True

        col_str = str(col_name).strip()
        raw_lower = col_str.lower()
        col_norm = cls.normalize(col_str)

        # 1. Check explicit index-like name patterns
        if _INDEX_NAME_REGEX.search(raw_lower) or _INDEX_NAME_REGEX.search(col_norm):
            return True

        if raw_lower.startswith("unnamed"):
            return True

        # 2. Check if series is an obvious sequential/index-like numeric column
        if series is not None:
            try:
                # If legitimate business identifier, do not classify as index solely due to values
                if cls.is_business_identifier(col_str):
                    return False

                # Drop nulls to inspect sequence
                if hasattr(series, "dropna"):
                    s = series.dropna()
                else:
                    s = series

                if len(s) >= 2:
                    # Check if numeric integer-like
                    is_numeric = False
                    if hasattr(pd.api.types, "is_numeric_dtype") and pd.api.types.is_numeric_dtype(s):
                        if not pd.api.types.is_bool_dtype(s):
                            is_numeric = True

                    if is_numeric:
                        try:
                            is_whole = (s % 1 == 0).all()
                        except Exception:
                            is_whole = False

                        if is_whole:
                            # Check if strictly increasing with step 1
                            diffs = s.diff().iloc[1:]
                            if (diffs == 1).all():
                                first_val = s.iloc[0]
                                # 0-indexed (0, 1, 2, ... N-1): classic dataframe index
                                if first_val == 0:
                                    return True
                                # 1-indexed (1, 2, 3, ... N): row counter without business identifier
                                if first_val == 1 and not cls.is_business_identifier(col_str):
                                    return True
            except Exception:
                pass

        return False


    @classmethod
    def split_camel_case(cls, s: str) -> str:
        """Splits camelCase or PascalCase into space-separated tokens."""
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", s)
        return s

    @classmethod
    def tokenize(cls, col_name: str) -> List[str]:
        """Deconstructs a column name into cleaned, canonical semantic tokens."""
        if not col_name:
            return []

        # 1. Split camelCase
        expanded = cls.split_camel_case(str(col_name))

        # 2. Replace all punctuation and separators with spaces
        cleaned = re.sub(r"[\W_]+", " ", expanded).strip().lower()

        raw_tokens = cleaned.split()
        normalized_tokens = []

        for token in raw_tokens:
            # Map known abbreviation if present
            norm_token = _SEMANTIC_ALIASES.get(token, token)
            # Check combined patterns like "custid"
            if norm_token.endswith("id") and len(norm_token) > 2 and norm_token != "id":
                prefix = norm_token[:-2]
                prefix_norm = _SEMANTIC_ALIASES.get(prefix, prefix)
                normalized_tokens.append(prefix_norm)
                normalized_tokens.append("id")
            elif norm_token.endswith("dt") and len(norm_token) > 2:
                prefix = norm_token[:-2]
                prefix_norm = _SEMANTIC_ALIASES.get(prefix, prefix)
                normalized_tokens.append(prefix_norm)
                normalized_tokens.append("date")
            elif norm_token.endswith("no") and len(norm_token) > 2:
                prefix = norm_token[:-2]
                prefix_norm = _SEMANTIC_ALIASES.get(prefix, prefix)
                normalized_tokens.append(prefix_norm)
                normalized_tokens.append("number")
            else:
                normalized_tokens.append(norm_token)

        return normalized_tokens

    @classmethod
    def normalize(cls, col_name: str) -> str:
        """
        Returns a canonical normalized string for a column name.
        Example: 'Customer ID' -> 'customer_id', 'cust-id' -> 'customer_id'.
        """
        tokens = cls.tokenize(col_name)
        if not tokens:
            return ""
        return "_".join(tokens)

    @classmethod
    def similarity(cls, col_a: str, col_b: str) -> float:
        """
        Calculates deterministic token-level Jaccard similarity between two column names.
        Returns float between 0.0 and 1.0.
        """
        norm_a = cls.normalize(col_a)
        norm_b = cls.normalize(col_b)

        if norm_a == norm_b and norm_a:
            return 1.0

        tokens_a = set(cls.tokenize(col_a))
        tokens_b = set(cls.tokenize(col_b))

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return len(intersection) / len(union)
