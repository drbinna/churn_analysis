import pandas as pd
import numpy as np

def generate_customer_data(n_customers=1000, random_state=42):
    """Generate synthetic customer data for churn analysis."""
    np.random.seed(random_state)
    
    customer_data = {
        'CustomerID': range(1, n_customers + 1),
        'Age': np.random.randint(18, 80, n_customers),
        'MonthlyCharge': np.round(np.random.uniform(20, 150, n_customers), 2),
        'CustomerServiceCalls': np.random.randint(0, 10, n_customers),
    }
    
    df = pd.DataFrame(customer_data)
    
    # Create realistic churn patterns
    churn_prob = np.zeros(n_customers)
    for i in range(n_customers):
        prob = 0.1  # Base probability
        
        # Age factor
        if df.loc[i, 'Age'] < 30 or df.loc[i, 'Age'] > 60:
            prob += 0.2
        
        # Monthly charge factor
        if df.loc[i, 'MonthlyCharge'] > 100:
            prob += 0.3
        
        # Customer service calls factor
        prob += df.loc[i, 'CustomerServiceCalls'] * 0.08
        
        churn_prob[i] = min(prob, 0.9)  # Cap at 90%
    
    # Generate churn labels
    df['Churn'] = np.random.binomial(1, churn_prob, n_customers)
    df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
    
    return df
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd

class ChurnPredictor:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced'
        )
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X, y):
        """Train the churn prediction model."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Make churn predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get churn probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save trained model."""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """Load saved model."""
        self.model = joblib.load(filepath)
        self.is_trained = True