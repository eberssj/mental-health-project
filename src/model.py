from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

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
    Creates an ethical Pipeline embedding PCA, SMOTE, and the Classifier.
    Prevents data leakage during cross-validation.
    """
    # Build pipeline: PCA reduces complexity -> SMOTE balances classes -> Classifier learns
    pipeline = ImbPipeline([
        ('pca', PCA(n_components=0.95, random_state=42)), # Keeps 95% of variance
        ('smote', SMOTE(random_state=42)),
        ('classifier', classifier)
    ])
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def train_nlp_xgb(X_train, y_train, X_val, y_val):
    """Optimized XGBoost Training for Text Data."""
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model