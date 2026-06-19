from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

def get_search_spaces():
    """Defines classifiers and hyperparameter spaces for evaluation."""
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, solver='saga'),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss', tree_method='hist')
    }
    
    param_grids = {
        'LogisticRegression': {
            'classifier__C': [0.1, 1.0, 10.0]
        },
        'RandomForest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [4, 6]
        }
    }
    return classifiers, param_grids

def train_and_tune_structured(X_train, y_train, model_name, classifier, param_grid):
    """
    Creates a Pipeline with PCA and the classifier.
    The structured dataset is already close to balanced, so no synthetic
    oversampling is applied.
    """
    pipeline = Pipeline([
        ('pca', PCA(n_components=0.95, random_state=42)), # Keeps 95% of variance
        ('classifier', classifier)
    ])
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def train_nlp_xgb(X_train, y_train, X_val, y_val):
    """Optimized XGBoost Training for Text Data using balanced sample weights."""
    model = XGBClassifier(
        n_estimators=250, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42, 
        eval_metric='mlogloss',
        tree_method='hist',
        early_stopping_rounds=10,
        n_jobs=-1
    )
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def train_semantic_classifier(X_train, y_train):
    """
    Trains a balanced classifier over Sentence-BERT embeddings.

    The embeddings already carry semantic information, so a simpler linear
    classifier is a strong and interpretable baseline.
    """
    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
