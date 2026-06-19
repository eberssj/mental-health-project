import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import export_metrics, save_confusion_matrix, save_correlation_matrix
from src.model import (
    get_search_spaces,
    train_and_tune_structured,
    train_nlp_xgb,
    train_semantic_classifier,
)
from src.preprocessing import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    get_embedding_cache_path,
    get_semantic_embeddings,
    get_tfidf_vectorizer,
    load_structured_data,
    process_text_data,
)


def print_progress(step, total, message):
    percent = (step / total) * 100
    bar = "#" * int(step * 25 / total) + "-" * (25 - int(step * 25 / total))
    sys.stdout.write(f"\rProgress: |{bar}| {percent:.1f}% - {message}")
    sys.stdout.flush()


def create_folders():
    for path in ["exports/models", "exports/plots", "exports/embeddings"]:
        os.makedirs(path, exist_ok=True)


def run_structured_pipeline():
    print("\n\n--- [PART 1] STRUCTURED SYMPTOMS PIPELINE (MULTI-MODEL) ---")
    create_folders()

    print("Loading illness_dataset.csv...")
    X, y, le = load_structured_data("data/illness_dataset.csv")
    print("Class distribution (no resampling applied):")
    print(pd.Series(le.inverse_transform(y)).value_counts().sort_index())

    print("Generating Psychiatric Correlation Matrix...")
    save_correlation_matrix(X, "structured_correlation")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    classifiers, param_grids = get_search_spaces()
    results = {}
    best_overall_score = -1
    best_overall_pipeline = None

    total_models = len(classifiers)

    for idx, (name, clf) in enumerate(classifiers.items(), 1):
        print_progress(idx, total_models, f"Tuning & Validating {name}...")

        # GridSearch trains using Pipeline (PCA -> Model) without synthetic oversampling.
        best_pipe, best_params, cv_score = train_and_tune_structured(
            X_train,
            y_train,
            name,
            clf,
            param_grids[name]
        )

        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)

        print(f"\n\nResults for {name}:")
        export_metrics(y_test, y_pred, y_prob, le.classes_, name, best_params)
        save_confusion_matrix(y_test, y_pred, le.classes_, f"cm_structured_{name}")

        results[name] = cv_score

        if cv_score > best_overall_score:
            best_overall_score = cv_score
            best_overall_pipeline = best_pipe

    print("\n" + "=" * 40)
    print("STRUCTURED BENCHMARK COMPARISON")
    print("=" * 40)
    for model_name, score in results.items():
        print(f"{model_name}: Mean CV F1-macro = {score:.4f}")
    print("=" * 40)

    joblib.dump(best_overall_pipeline, "exports/models/best_structured_pipeline.pkl")
    joblib.dump(le, "exports/models/structured_encoder.pkl")
    print("OK - Best Structured Pipeline Saved Successfully.")


def run_nlp_pipeline():
    print("\n--- [PART 2] NLP PIPELINE (TF-IDF BASELINE + SEMANTIC EMBEDDINGS) ---")
    create_folders()

    texts, y, le_nlp = process_text_data(
        "data/Mental Health Disorder Detection Dataset.csv"
    )
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n[2.1] Training TF-IDF + XGBoost baseline...")
    vectorizer = get_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_t)
    X_test_vec = vectorizer.transform(X_test_t)
    tfidf_model = train_nlp_xgb(X_train_vec, y_train_t, X_test_vec, y_test_t)

    y_pred = tfidf_model.predict(X_test_vec)
    y_prob = tfidf_model.predict_proba(X_test_vec)

    tfidf_report = export_metrics(
        y_test_t,
        y_pred,
        y_prob,
        le_nlp.classes_,
        "TF-IDF + XGBoost NLP"
    )
    save_confusion_matrix(y_test_t, y_pred, le_nlp.classes_, "cm_nlp_tfidf_xgb")

    joblib.dump(tfidf_model, "exports/models/nlp_model.pkl")
    joblib.dump(tfidf_model, "exports/models/nlp_tfidf_xgb_model.pkl")
    joblib.dump(vectorizer, "exports/models/nlp_vectorizer.pkl")
    joblib.dump(vectorizer, "exports/models/nlp_tfidf_vectorizer.pkl")

    print("\n[2.2] Generating local Hugging Face semantic embeddings...")
    embedding_model_name = DEFAULT_SENTENCE_TRANSFORMER_MODEL
    print(f"Embedding model: {embedding_model_name}")
    train_cache = get_embedding_cache_path(X_train_t, "train", embedding_model_name)
    test_cache = get_embedding_cache_path(X_test_t, "test", embedding_model_name)
    X_train_emb = get_semantic_embeddings(
        X_train_t,
        model_name=embedding_model_name,
        cache_path=train_cache,
        batch_size=32
    )
    X_test_emb = get_semantic_embeddings(
        X_test_t,
        model_name=embedding_model_name,
        cache_path=test_cache,
        batch_size=32
    )

    print("Training Sentence-BERT + Logistic Regression model...")
    semantic_model = train_semantic_classifier(X_train_emb, y_train_t)
    y_pred_semantic = semantic_model.predict(X_test_emb)
    y_prob_semantic = semantic_model.predict_proba(X_test_emb)

    semantic_report = export_metrics(
        y_test_t,
        y_pred_semantic,
        y_prob_semantic,
        le_nlp.classes_,
        "Sentence-BERT + Logistic Regression NLP"
    )
    save_confusion_matrix(
        y_test_t,
        y_pred_semantic,
        le_nlp.classes_,
        "cm_nlp_sentence_bert"
    )

    joblib.dump(semantic_model, "exports/models/nlp_semantic_classifier.pkl")
    joblib.dump(
        {
            "embedding_model_name": embedding_model_name,
            "normalize_embeddings": True,
            "train_cache": train_cache,
            "test_cache": test_cache
        },
        "exports/models/nlp_semantic_config.pkl"
    )
    joblib.dump(le_nlp, "exports/models/nlp_encoder.pkl")

    print("\nNLP MACRO F1 COMPARISON")
    print(f"TF-IDF + XGBoost: {tfidf_report.loc['macro avg', 'f1-score']:.4f}")
    print(
        "Sentence-BERT + Logistic Regression: "
        f"{semantic_report.loc['macro avg', 'f1-score']:.4f}"
    )
    print("OK - NLP Analysis Complete.")


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    valid_modes = {"all", "structured", "nlp"}
    if mode not in valid_modes:
        valid = ", ".join(sorted(valid_modes))
        raise SystemExit(f"Invalid mode '{mode}'. Use one of: {valid}")

    if mode in {"all", "structured"}:
        run_structured_pipeline()
    if mode in {"all", "nlp"}:
        run_nlp_pipeline()

    print("\nALL PROCESSES FINISHED EXECUTING.")


if __name__ == "__main__":
    main()
