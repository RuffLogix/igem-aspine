import pandas as pd
from utils import encode_sequence
from xpresso_module.model import XPressoModel

# Define feature columns and label
FEATURES = [
    "utr5_length", "cds_length", "intron_length", "utr3_length",
    "utr5_gc", "cds_gc", "utr3_gc", "orf_exon_density"
]
LABEL = ["MPE"]

# Load datasets
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    return df

train_df = load_and_clean_data("./data/train_clean.csv")
test_df = load_and_clean_data("./data/test_clean.csv")
valid_df = load_and_clean_data("./data/valid_clean.csv")

# Prepare feature and label data
def prepare_data(df, features, label):
    X = df[features + ["SEQ"]]
    y = df[label].to_numpy()
    promoters = encode_sequence(X["SEQ"].to_numpy())
    halflife = X[features].to_numpy()
    return promoters, halflife, y

train_promoter, train_halflife, train_mpe = prepare_data(train_df, FEATURES, LABEL)
test_promoter, test_halflife, test_mpe = prepare_data(test_df, FEATURES, LABEL)
valid_promoter, valid_halflife, valid_mpe = prepare_data(valid_df, FEATURES, LABEL)

# Print shapes for validation
def print_data_shapes(dataset_name, promoter, halflife, mpe):
    print(f"=== {dataset_name} ===")
    print(f"Promoter: {promoter.shape}")
    print(f"Halflife: {halflife.shape}")
    print(f"MPE: {mpe.shape}")

print_data_shapes("Train", train_promoter, train_halflife, train_mpe)
print_data_shapes("Test", test_promoter, test_halflife, test_mpe)
print_data_shapes("Valid", valid_promoter, valid_halflife, valid_mpe)


# Model creation and loading
model = XPressoModel(
    promoter_shape=valid_promoter.shape[1:],
    halflife_shape=valid_halflife.shape[1:]
)

# Model Loading
# model.load("./model_weight.keras")
# print("Model loaded successfully")

# Model training
result = model.fit(
    valid_promoter, valid_halflife, valid_mpe,
    valid_promoter, valid_halflife, valid_mpe,
    n_epochs=1
)
print("Training completed successfully")

# Model prediction
predictions = model(valid_promoter, valid_halflife)
print("Predictions completed successfully")

# Display results
print(result)
