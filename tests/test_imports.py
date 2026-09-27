import importlib
import unittest


class TestPackageImports(unittest.TestCase):
    def test_project_packages_are_importable(self) -> None:
        package_names = (
            "recsys",
            "recsys.models",
            "recsys.models.baselines",
            "recsys.data",
            "recsys.training",
            "recsys.evaluation",
        )

        for package_name in package_names:
            with self.subTest(package=package_name):
                module = importlib.import_module(package_name)
                self.assertEqual(module.__name__, package_name)


if __name__ == "__main__":
    unittest.main()
