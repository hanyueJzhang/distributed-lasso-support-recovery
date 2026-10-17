# Data Contract

The original course dataset is intentionally excluded from this repository.

To use authorized data with the implementation, prepare one training matrix and response vector per node. All nodes must share the same ordered feature set.

Expected shapes:

- `X_train`: `(number_of_training_rows, number_of_features)`
- `y_train`: `(number_of_training_rows,)`
- `X_validation`: `(number_of_validation_rows, number_of_features)`
- `y_validation`: `(number_of_validation_rows,)`

Recommended preparation:

1. Remove identifiers and fields not required for modeling.
2. Standardize features using statistics computed only from the corresponding training partition.
3. Apply the same feature order to every node and validation set.
4. Keep raw data outside version control.
5. Document the source, license, and permitted use of any replacement dataset.

Run `examples/run_synthetic_demo.py` for a data-free example.

