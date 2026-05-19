import os
import joblib
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

from src.preprocessing import load_structured_data, process_text_data, get_tfidf_vectorizer
from src.model import get_search_spaces, train_and_tune_structured, train_nlp_xgb
from src.evaluation import export_metrics, save_confusion_matrix, save_correlation_matrix

def print_progress(step, total, message):
    percent = (step / total) * 100
    bar = '█' * int(step * 25 / total) + '-' * (25 - int(step * 25 / total))
    sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}% - {message}')
    sys.stdout.flush()

def create_folders():
    for path in ['exports/models', 'exports/plots']:
        os.makedirs(path, exist_ok=True)

def run_structured_pipeline():
    print("\n\n--- [PART 1] STRUCTURED SYMPTOMS PIPELINE (MULTI-MODEL) ---")
    create_folders()
    
    print("Loading illness_dataset.csv...")
    X, y, le = load_structured_data('data/illness_dataset.csv')
    
    print("Generating Psychiatric Correlation Matrix...")
    save_correlation_matrix(X, "structured_correlation")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    classifiers, param_grids = get_search_spaces()
    results = {}
    best_overall_score = 0
    best_overall_pipeline = None
    
    total_models = len(classifiers)
    
    for idx, (name, clf) in enumerate(classifiers.items(), 1):
        print_progress(idx, total_models, f"Tuning & Validating {name}...")
        
        # GridSearch trains using ImbPipeline (PCA -> SMOTE -> Model) avoiding Leakage
        best_pipe, best_params, cv_score = train_and_tune_structured(
            X_train, y_train, name, clf, param_grids[name]
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

    print("\n" + "="*40)
    print("STRUCTURED BENCHMARK COMPARISON")
    print("="*40)
    for model_name, score in results.items():
        print(f"{model_name}: Mean CV Accuracy = {score:.4f}")
    print("="*40)
    
    # Save the absolute best model architecture found
    joblib.dump(best_overall_pipeline, 'exports/models/best_structured_pipeline.pkl')
    joblib.dump(le, 'exports/models/structured_encoder.pkl')
    print("✅ Best Structured Pipeline Saved Successfully.")

def run_nlp_pipeline():
    print("\n--- [PART 2] OPTIMIZED NLP PIPELINE ---")
    
    texts, y, le_nlp = process_text_data('data/Mental Health Disorder Detection Dataset.csv')
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Vectorizing Text Features...")
    vectorizer = get_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_t)
    X_test_vec = vectorizer.transform(X_test_t)
    
    print("Training Optimized XGBoost NLP Model...")
    nlp_model = train_nlp_xgb(X_train_vec, y_train_t, X_test_vec, y_test_t)
    
    y_pred = nlp_model.predict(X_test_vec)
    y_prob = nlp_model.predict_proba(X_test_vec)
    
    export_metrics(y_test_t, y_pred, y_prob, le_nlp.classes_, "XGBoost NLP")
    save_confusion_matrix(y_test_t, y_pred, le_nlp.classes_, "cm_nlp")
    
    joblib.dump(nlp_model, 'exports/models/nlp_model.pkl')
    joblib.dump(vectorizer, 'exports/models/nlp_vectorizer.pkl')
    joblib.dump(le_nlp, 'exports/models/nlp_encoder.pkl')
    print("✅ NLP Analysis Complete.")

if __name__ == "__main__":
    run_structured_pipeline()
    run_nlp_pipeline()
    print("\n✨ ALL PROCESSES FINISHED EXECUTING.")