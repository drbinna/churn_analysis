# Customer Churn Analysis - Standalone Python Script
# No Jupyter required - runs directly in Python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_directories():
    """Create necessary directories for outputs"""
    directories = [
        'data/processed',
        'results/figures', 
        'results/reports',
        'models/trained'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Output directories created")

def generate_customer_data(n_customers=1000):
    """Generate synthetic customer data for churn analysis"""
    print(f"📊 Generating {n_customers} synthetic customer records...")
    
    # Generate basic customer data
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
        
        # Age factor - younger and older customers more likely to churn
        if df.loc[i, 'Age'] < 30 or df.loc[i, 'Age'] > 60:
            prob += 0.2
        
        # Monthly charge factor - expensive plans increase churn
        if df.loc[i, 'MonthlyCharge'] > 100:
            prob += 0.3
        
        # Customer service calls - more calls = higher churn risk
        prob += df.loc[i, 'CustomerServiceCalls'] * 0.08
        
        churn_prob[i] = min(prob, 0.9)  # Cap at 90%
    
    # Generate churn labels based on probabilities
    df['Churn'] = np.random.binomial(1, churn_prob, n_customers)
    df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
    
    print(f"✅ Dataset created with {len(df)} customers")
    print(f"📈 Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print(f"\n📏 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🔍 Missing values: {df.isnull().sum().sum()}")
    
    print(f"\n📊 Churn Distribution:")
    churn_counts = df['Churn'].value_counts()
    for status, count in churn_counts.items():
        percentage = count / len(df) * 100
        print(f"  {status}: {count} customers ({percentage:.1f}%)")
    
    print(f"\n📈 Statistical Summary:")
    print(df.describe())
    
    return df

def train_model(df):
    """Train the churn prediction model"""
    print("\n" + "="*60)
    print("🌳 MODEL TRAINING")
    print("="*60)
    
    # Prepare features and target
    X = df.drop(['CustomerID', 'Churn'], axis=1)
    y = df['Churn']
    
    print(f"🎯 Features: {list(X.columns)}")
    print(f"🎯 Target: Churn prediction")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🚂 Training set: {X_train.shape[0]} samples")
    print(f"🧪 Testing set: {X_test.shape[0]} samples")
    
    # Create and train model
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced'
    )
    
    print("\n🔄 Training Decision Tree model...")
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("📈 MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    
    # Classification report
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(cm)
    
    return y_pred, y_pred_proba, accuracy, cm

def analyze_feature_importance(model, feature_names):
    """Analyze and display feature importance"""
    print("\n" + "="*60)
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("📊 Feature Importance Ranking:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance']:.1%})")
    
    return feature_importance

def create_visualizations(df, model, feature_importance, cm, y_test, y_pred):
    """Create and save visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Customer Churn Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Churn Distribution (Pie Chart)
    churn_counts = df['Churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for No, Red for Yes
    axes[0, 0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('🎯 Churn Distribution')
    
    # 2. Feature Importance
    bars = axes[0, 1].bar(feature_importance['Feature'], feature_importance['Importance'], 
                         color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 1].set_title('🏆 Feature Importance')
    axes[0, 1].set_xlabel('Features')
    axes[0, 1].set_ylabel('Importance Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'], ax=axes[1, 0])
    axes[1, 0].set_title('🔥 Confusion Matrix')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xlabel('Predicted')
    
    # 4. Age vs Monthly Charge Scatter Plot
    scatter = axes[1, 1].scatter(df['Age'], df['MonthlyCharge'], 
                                c=df['Churn'].map({'No': 0, 'Yes': 1}), 
                                cmap='RdYlBu_r', alpha=0.6, s=30)
    axes[1, 1].set_title('🔄 Age vs Monthly Charge (Colored by Churn)')
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('Monthly Charge ($)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1], ticks=[0, 1])
    cbar.set_label('Churn (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.savefig('results/figures/churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Dashboard saved to: results/figures/churn_analysis_dashboard.png")
    
    # Create individual plots
    create_individual_plots(df, feature_importance)

def create_individual_plots(df, feature_importance):
    """Create individual visualization plots"""
    
    # Age distribution by churn
    plt.figure(figsize=(10, 6))
    for churn_status in ['No', 'Yes']:
        ages = df[df['Churn'] == churn_status]['Age']
        plt.hist(ages, alpha=0.7, label=f'Churn: {churn_status}', bins=15)
    plt.title('👥 Age Distribution by Churn Status')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('results/figures/age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Monthly charge distribution
    plt.figure(figsize=(10, 6))
    for churn_status in ['No', 'Yes']:
        charges = df[df['Churn'] == churn_status]['MonthlyCharge']
        plt.hist(charges, alpha=0.7, label=f'Churn: {churn_status}', bins=15)
    plt.title('💰 Monthly Charge Distribution by Churn Status')
    plt.xlabel('Monthly Charge ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('results/figures/monthly_charge_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Individual plots saved to results/figures/")

def save_results(df, model, feature_importance, accuracy):
    """Save analysis results to files"""
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    # Save datasets
    df.to_csv('data/processed/customer_data_with_churn.csv', index=False)
    print("✅ Customer data saved to: data/processed/customer_data_with_churn.csv")
    
    # Save feature importance
    feature_importance.to_csv('results/feature_importance.csv', index=False)
    print("✅ Feature importance saved to: results/feature_importance.csv")
    
    # Save model summary
    model_summary = {
        'model_type': 'DecisionTreeClassifier',
        'accuracy': accuracy,
        'parameters': {
            'max_depth': 5,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'class_weight': 'balanced'
        },
        'feature_importance': feature_importance.to_dict('records')
    }
    
    import json
    with open('results/model_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=2)
    print("✅ Model summary saved to: results/model_summary.json")
    
    # Save the trained model
    import joblib
    joblib.dump(model, 'models/trained/churn_decision_tree.pkl')
    print("✅ Trained model saved to: models/trained/churn_decision_tree.pkl")

def generate_business_insights(feature_importance, accuracy):
    """Generate business insights and recommendations"""
    print("\n" + "="*60)
    print("💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    print("\n🔍 KEY FINDINGS:")
    print("-" * 50)
    
    top_feature = feature_importance.iloc[0]
    print(f"🏆 Most Important Factor: {top_feature['Feature']} ({top_feature['Importance']:.1%})")
    print(f"🎯 Model Accuracy: {accuracy:.1%}")
    
    print(f"\n📊 Feature Ranking:")
    for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
        print(f"{i}. {row['Feature']}: {row['Importance']:.3f}")
    
    print(f"\n💼 ACTIONABLE RECOMMENDATIONS:")
    print("-" * 50)
    
    recommendations = [
        "📞 CUSTOMER SERVICE OPTIMIZATION:",
        "   • Monitor customers with >3 service calls closely",
        "   • Implement proactive support for frequent callers",
        "   • Create escalation procedures for service issues",
        "",
        "💰 PRICING STRATEGY:",
        "   • Review pricing for customers paying >$100/month",
        "   • Offer loyalty discounts for high-value customers",
        "   • Create value-added packages to justify pricing",
        "",
        "🎯 RETENTION CAMPAIGNS:",
        "   • Focus on customers aged 18-30 and 60+",
        "   • Develop age-specific retention strategies",
        "   • Create targeted offers for at-risk segments",
        "",
        "📊 MONITORING & ALERTS:",
        "   • Use this model for monthly customer scoring",
        "   • Set up automated alerts for high-risk customers",
        "   • Track intervention success rates"
    ]
    
    for rec in recommendations:
        print(rec)

def simulate_customer_scoring():
    """Simulate customer risk scoring for new customers"""
    print("\n" + "="*60)
    print("🚀 CUSTOMER RISK SCORING SIMULATION")
    print("="*60)
    
    # Load the trained model
    import joblib
    model = joblib.load('models/trained/churn_decision_tree.pkl')
    
    # Create sample customers
    new_customers = pd.DataFrame({
        'Age': [25, 45, 65, 35, 55],
        'MonthlyCharge': [120, 80, 95, 140, 70],
        'CustomerServiceCalls': [5, 1, 3, 7, 0]
    })
    
    # Predict churn probabilities
    churn_probabilities = model.predict_proba(new_customers)[:, 1]
    predictions = model.predict(new_customers)
    
    # Assign risk levels
    def get_risk_level(prob):
        if prob > 0.7:
            return "🔴 HIGH RISK"
        elif prob > 0.4:
            return "🟡 MEDIUM RISK"
        else:
            return "🟢 LOW RISK"
    
    risk_levels = [get_risk_level(p) for p in churn_probabilities]
    
    print("🎯 SAMPLE CUSTOMER RISK ASSESSMENT:")
    print("-" * 50)
    
    for i, (_, customer) in enumerate(new_customers.iterrows()):
        print(f"\n👤 Customer {i+1}:")
        print(f"   Age: {customer['Age']} years")
        print(f"   Monthly Charge: ${customer['MonthlyCharge']:.2f}")
        print(f"   Service Calls: {customer['CustomerServiceCalls']}")
        print(f"   Churn Probability: {churn_probabilities[i]:.2%}")
        print(f"   Prediction: {predictions[i]}")
        print(f"   Risk Level: {risk_levels[i]}")

def main():
    """Main analysis function"""
    print("🚀 CUSTOMER CHURN PREDICTION ANALYSIS")
    print("=" * 60)
    print("📍 Running standalone Python analysis")
    print("⏰ Starting analysis...")
    
    # Create output directories
    create_directories()
    
    # Generate and explore data
    df = generate_customer_data(1000)
    df = explore_data(df)
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(df)
    
    # Evaluate model
    y_pred, y_pred_proba, accuracy, cm = evaluate_model(model, X_test, y_test)
    
    # Analyze features
    feature_names = ['Age', 'MonthlyCharge', 'CustomerServiceCalls']
    feature_importance = analyze_feature_importance(model, feature_names)
    
    # Create visualizations
    create_visualizations(df, model, feature_importance, cm, y_test, y_pred)
    
    # Save all results
    save_results(df, model, feature_importance, accuracy)
    
    # Generate insights
    generate_business_insights(feature_importance, accuracy)
    
    # Simulate scoring
    simulate_customer_scoring()
    
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📋 SUMMARY:")
    print(f"   • Model Accuracy: {accuracy:.2%}")
    print(f"   • Dataset Size: {len(df):,} customers")
    print(f"   • Churn Rate: {(df['Churn'] == 'Yes').mean():.2%}")
    print(f"   • Most Important Factor: {feature_importance.iloc[0]['Feature']}")
    
    print(f"\n📁 OUTPUT FILES CREATED:")
    print(f"   • Customer data: data/processed/customer_data_with_churn.csv")
    print(f"   • Visualizations: results/figures/churn_analysis_dashboard.png")
    print(f"   • Feature importance: results/feature_importance.csv")
    print(f"   • Trained model: models/trained/churn_decision_tree.pkl")
    print(f"   • Model summary: results/model_summary.json")
    
    print(f"\n✅ Your churn prediction analysis is complete!")
    print(f"🔗 All files are ready for your GitHub repository")

if __name__ == "__main__":
    main()