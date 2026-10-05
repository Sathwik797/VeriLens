import ast
import inspect
import unittest
import pandas as pd

from app.executor.safe_executor import (
    ExecutionResult,
    ExecutorColumnError,
    ExecutorError,
    ExecutorOperationError,
    SafeExecutor,
)
from app.models.analysis_plan import AnalysisPlan, FilterCondition


class TestSafeExecutor(unittest.TestCase):

    def setUp(self):
        # Deterministic sample dataset mimicking e-commerce store
        self.df = pd.DataFrame({
            "Region": ["West", "East", "West", "Central", "East", "South"],
            "Category": ["Technology", "Furniture", "Technology", "Office", "Technology", "Furniture"],
            "Sales": [500.0, 200.0, 300.0, 150.0, 400.0, 100.0],
            "Profit": [120.0, 30.0, 80.0, 25.0, 90.0, 10.0],
        })

    def test_01_groupby_sum(self):
        """Test: Groupby Region and calculate sum of Profit."""
        plan = AnalysisPlan(
            operation="groupby",
            group_by="Region",
            metric="Profit",
            aggregation="sum",
            sort="descending",
        )
        res = SafeExecutor.execute(self.df, plan)

        self.assertIsInstance(res, ExecutionResult)
        records = res.to_records()
        # West: 120 + 80 = 200.0, East: 30 + 90 = 120.0, Central: 25.0, South: 10.0
        self.assertEqual(records[0]["Region"], "West")
        self.assertEqual(records[0]["Profit"], 200.0)
        self.assertEqual(records[1]["Region"], "East")
        self.assertEqual(records[1]["Profit"], 120.0)

    def test_02_groupby_mean(self):
        """Test: Groupby Category and calculate mean of Sales."""
        plan = AnalysisPlan(
            operation="groupby",
            group_by="Category",
            metric="Sales",
            aggregation="mean",
            sort="descending",
        )
        res = SafeExecutor.execute(self.df, plan)
        records = res.to_records()
        # Technology: (500 + 300 + 400)/3 = 400.0
        # Furniture: (200 + 100)/2 = 150.0
        # Office: 150.0
        self.assertEqual(records[0]["Category"], "Technology")
        self.assertEqual(records[0]["Sales"], 400.0)

    def test_03_aggregate_scalar(self):
        """Test: Dataset-wide aggregation (total sales sum)."""
        plan = AnalysisPlan(
            operation="aggregate",
            metric="Sales",
            aggregation="sum",
        )
        res = SafeExecutor.execute(self.df, plan)
        # Total sales = 500 + 200 + 300 + 150 + 400 + 100 = 1650.0
        self.assertEqual(res.scalar_value, 1650.0)

    def test_04_filter_and_aggregate(self):
        """Test: Filter to Technology category, calculate mean profit."""
        plan = AnalysisPlan(
            operation="aggregate",
            metric="Profit",
            aggregation="mean",
            filters=[FilterCondition(column="Category", operator="==", value="Technology")],
        )
        res = SafeExecutor.execute(self.df, plan)
        # Technology profits: 120, 80, 90 -> Mean = 96.666...
        self.assertAlmostEqual(res.scalar_value, 96.66666666666667)

    def test_05_sorting_ascending_descending(self):
        """Test: Sort operation by Sales."""
        plan_desc = AnalysisPlan(
            operation="sort",
            metric="Sales",
            sort="descending",
        )
        res_desc = SafeExecutor.execute(self.df, plan_desc)
        records_desc = res_desc.to_records()
        self.assertEqual(records_desc[0]["Sales"], 500.0)
        self.assertEqual(records_desc[-1]["Sales"], 100.0)

        plan_asc = AnalysisPlan(
            operation="sort",
            metric="Sales",
            sort="ascending",
        )
        res_asc = SafeExecutor.execute(self.df, plan_asc)
        records_asc = res_asc.to_records()
        self.assertEqual(records_asc[0]["Sales"], 100.0)
        self.assertEqual(records_asc[-1]["Sales"], 500.0)

    def test_06_value_counts(self):
        """Test: Value counts on Category column."""
        plan = AnalysisPlan(
            operation="value_counts",
            metric="Category",
            sort="descending",
        )
        res = SafeExecutor.execute(self.df, plan)
        records = res.to_records()
        # Technology appears 3 times, Furniture 2 times, Office 1 time
        self.assertEqual(records[0]["Category"], "Technology")
        self.assertEqual(records[0]["count"], 3)
        self.assertEqual(records[1]["Category"], "Furniture")
        self.assertEqual(records[1]["count"], 2)

    def test_07_limit_top_n(self):
        """Test: Groupby with limit 1 (Which region has highest profit)."""
        plan = AnalysisPlan(
            operation="groupby",
            group_by=["Region"],
            metric="Profit",
            aggregation="sum",
            sort="descending",
            limit=1,
        )
        res = SafeExecutor.execute(self.df, plan)
        records = res.to_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["Region"], "West")
        self.assertEqual(records[0]["Profit"], 200.0)

    def test_08_invalid_column_rejection(self):
        """Test: Attempting to reference a non-existent column raises ExecutorColumnError."""
        plan = AnalysisPlan(
            operation="groupby",
            group_by="NonExistentColumn",
            metric="Profit",
            aggregation="sum",
        )
        with self.assertRaises(ExecutorColumnError) as ctx:
            SafeExecutor.execute(self.df, plan)
        self.assertIn("NonExistentColumn", str(ctx.exception))

    def test_09_unsupported_operation_rejection(self):
        """Test: Plan with missing required fields or invalid configuration raises error."""
        # groupby without group_by field
        plan = AnalysisPlan.model_construct(
            operation="groupby",
            group_by=None,
            metric="Profit",
        )
        with self.assertRaises(ExecutorOperationError):
            SafeExecutor.execute(self.df, plan)

    def test_10_no_eval_or_exec_in_codebase(self):
        """Security Test: Statically inspect safe_executor.py AST to guarantee no eval or exec is used."""
        import app.executor.safe_executor as safe_module
        source = inspect.getsource(safe_module)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertNotIn(
                        node.func.id,
                        {"eval", "exec", "__import__", "compile"},
                        f"Prohibited function '{node.func.id}' found in safe_executor.py!"
                    )


if __name__ == "__main__":
    unittest.main()
