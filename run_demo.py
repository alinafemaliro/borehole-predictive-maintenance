import numpy as np
import yaml

print("run_demo.py started")

from src.modeling.model_comparator import ModelComparator
print("Imported ModelComparator")

def main():
    print("Loading config...")
    with open("config/model_comparison.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Config loaded. Models:", config["model_comparison"]["models_to_compare"])

    np.random.seed(42)
    X_train = np.random.randn(300, 10)
    y_train = (np.random.rand(300) > 0.85).astype(int)

    X_test = np.random.randn(120, 10)
    y_test = (np.random.rand(120) > 0.85).astype(int)

    print("Data ready. Running comparison...")
    comparator = ModelComparator(config)

    results = comparator.run_comparison(X_train, y_train, X_test, y_test)

    print("Done. Best models:")
    print(results["best_models"])

if __name__ == "__main__":
    main()
