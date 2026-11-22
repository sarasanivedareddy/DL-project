# Social Media Trend Analyzer

A desktop application that analyzes tweet text and predicts whether a tweet belongs to a **trend** or **no-trend** category. The application combines text preprocessing, a Bidirectional LSTM neural network, a Random Forest classifier, and an XGBoost baseline. It provides a Tkinter graphical interface for uploading datasets, preprocessing text, training/evaluating models, and testing new tweet data.

## Features

- Upload a labeled tweet dataset in CSV format.
- Clean tweet text by lowercasing, removing punctuation and non-alphabetic tokens, filtering English stop words, and lemmatizing words.
- Convert cleaned text into integer sequences using a Keras tokenizer and pad sequences to a fixed length of 100.
- Split the data into training and test sets (80/20, with `random_state=42`).
- Train a Bidirectional LSTM text model and use its intermediate representation as features for a Random Forest classifier.
- Train or load an XGBoost classifier.
- Evaluate models using accuracy, weighted precision, weighted recall, weighted F1-score, classification report, and confusion matrix.
- Predict trend labels for a separate CSV of tweet text.
- Display results in a Tkinter desktop interface and show confusion-matrix heatmaps.

## Models

### Bidirectional LSTM + Random Forest

The LSTM model uses an embedding layer (vocabulary size 10,000 and embedding dimension 50), a Bidirectional LSTM layer with 128 units, a 64-unit ReLU dense layer, and dropout. It is trained with Adam and binary cross-entropy. The output layer is excluded when extracting intermediate features; those features are then used to train a Random Forest classifier.

### XGBoost

The application also trains an `XGBClassifier` on padded token sequences and reports classification metrics. This provides an additional model path for comparison.

## Repository contents

```text
DL-project/
├── README.md
└── Socialmedia       # Tkinter application (Python source; currently no .py extension)
```

The application saves generated artifacts in the working directory, including `tokenizer.pickle`, `lstm_model.h5`, `random_forest_model.joblib`, and `model/xgb.pkl`.

## Requirements

Use a Python version supported by your installed TensorFlow release. Install the libraries used by the application:

```bash
python -m pip install pandas nltk scikit-learn tensorflow xgboost joblib seaborn matplotlib
```

Tkinter is included with many standard Python installations. On some Linux distributions it must be installed separately (for example, the `python3-tk` system package).

Download the NLTK resources used during preprocessing:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Dataset format

The training CSV is expected to contain these columns:

| Column | Description |
|---|---|
| `text` | Tweet or social-media post text |
| `target` | Binary class label used by the model |

The prediction CSV is expected to contain tweet text in its first column. Select the appropriate CSV file through the application's file dialog. Training and prediction files should follow these column conventions.

The source code refers to a `TwitterNewsData` directory as the initial file-dialog location. If that directory is not present, use the file dialog to navigate to the dataset location.

## Run the application

From the repository root:

```bash
python Socialmedia
```

The source file currently has no `.py` extension, but Python can execute it as shown. For clarity, it may be renamed to `Socialmedia.py`; if renamed, run:

```bash
python Socialmedia.py
```

## Application workflow

1. **Upload Trend Tweet Dataset** — select the labeled training CSV.
2. **Preprocess Dataset** — clean the tweet text, fit the tokenizer, create padded sequences, encode labels, and create the train/test split.
3. **Run LSTM Algorithm** — train or load the BiLSTM model, extract its learned text features, train/load the Random Forest classifier, and display evaluation metrics.
4. **Existing XGB Algorithm** — train or load the XGBoost model and display its evaluation metrics.
5. **Test Tweet Detection** — select a prediction CSV and classify its tweet text using the saved LSTM feature extractor and Random Forest model.

Run preprocessing before either training/evaluation button. Train or load the LSTM/Random Forest path before using the prediction button.

## Saved model and preprocessing files

| Artifact | Purpose |
|---|---|
| `tokenizer.pickle` | Vocabulary and token-to-index mapping fitted on training text |
| `lstm_model.h5` | Trained Bidirectional LSTM model |
| `random_forest_model.joblib` | Random Forest model trained on LSTM features |
| `model/xgb.pkl` | Saved XGBoost classifier |

Keep the tokenizer and corresponding models together: predictions must use the same tokenizer and preprocessing settings used during training. Do not load pickle/joblib files from untrusted sources.

The XGBoost code saves to a `model/` subdirectory. Create that directory in the project root before running the XGBoost option if it does not already exist:

```bash
mkdir model
```

## Evaluation

The application calculates:

- **Accuracy** — overall fraction of correct predictions.
- **Precision** — reliability of positive predictions.
- **Recall** — fraction of actual positives detected.
- **Weighted F1-score** — weighted balance of precision and recall.
- **Confusion matrix and classification report** — class-level performance details.

Metrics are printed to the console and displayed in the application; confusion matrices are visualized as heatmaps. No benchmark score is stated here because results depend on the selected dataset and trained model artifacts.

## Notes and limitations

- This is a binary classification prototype. Confirm the dataset's label definition before interpreting class values. The current prediction display maps class `0` to **TREND** and other predictions to **NO TREND**.
- The train/test split is not stratified in the current implementation.
- The tokenizer is fitted before the train/test split, so vocabulary construction sees all available text. For a leakage-resistant evaluation, split first and fit the tokenizer only on training text.
- The XGBoost path operates on padded token IDs, while the BiLSTM + Random Forest path uses learned sequence features. Their input representations differ.
- Verify the prediction path end to end after training: the saved LSTM, tokenizer, and Random Forest model must all be available and loaded before prediction.
- Model files are generated locally and should generally not be committed to Git.
- This README describes the current source and expected workflow; it does not imply that every runtime path has been independently tested.

## Possible future improvements

- Add a `requirements.txt` with tested dependency versions.
- Rename the source file to `Socialmedia.py` and separate GUI, preprocessing, training, and prediction code into modules.
- Add input validation for missing columns, empty text, and unexpected labels.
- Use a stratified train/test split and fit preprocessing only on training data.
- Persist and reload the Random Forest model consistently during prediction.
- Add automated tests and a reproducible experiment/results report.

## License

No license is specified in this repository at the time this README was prepared. Add a license file before redistributing or reusing the project.
