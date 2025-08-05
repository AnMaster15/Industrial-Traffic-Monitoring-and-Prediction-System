import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Removes rows with outliers using IQR method"""
    def fit(self, X, y=None):
        self.iqr_bounds = {}
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.iqr_bounds[col] = (lower_bound, upper_bound)
        return self

    def transform(self, X, y=None):
        mask = pd.Series(True, index=X.index)
        for col in X.columns:
            lb, ub = self.iqr_bounds[col]
            mask &= (X[col] >= lb) & (X[col] <= ub)
        return X[mask]

class TrafficFlowPredictor:
    """Predict future traffic flow using KNN and preprocessing"""

    def __init__(self, model_path='traffic_flow_model.pkl'):
        self.model_path = model_path
        if Path(model_path).exists():
            self.pipeline = joblib.load(model_path)
        else:
            self.pipeline = None

    def train(self, df: pd.DataFrame):
        """Train model on historical data"""
        df['target'] = df['weighted_total'].shift(-1).ffill()

        features = df[[
            'weighted_total', 'average_speed', 'congestion_level',
            'traffic_flow_efficiency', 'average_waiting_time'
        ]].copy()
        labels = df['target']

        valid = features.notnull().all(axis=1) & labels.notnull()
        features = features[valid]
        labels = labels[valid]

        if len(features) == 0:
            raise ValueError("No valid rows left to train on.")

        # ✅ Apply outlier removal BEFORE pipeline
        remover = OutlierRemover()
        features = remover.fit_transform(features)
        labels = labels.loc[features.index]  # Align labels to filtered features

        preprocess = Pipeline(steps=[
            ('scaling', StandardScaler())
        ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocess),
            ('model', KNeighborsRegressor(n_neighbors=5))
        ])

        self.pipeline.fit(features, labels)
        joblib.dump(self.pipeline, self.model_path)


    def predict(self, metrics: dict) -> float:
        """Predict traffic weighted total for next interval"""
        try:
            # Handle missing input values
            safe_metrics = {
                k: (0.0 if v is None or pd.isna(v) else v)
                for k, v in metrics.items()
            }
            feature_df = pd.DataFrame([safe_metrics])
            return self.pipeline.predict(feature_df)[0]
        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            return metrics.get('weighted_total', 0.0)