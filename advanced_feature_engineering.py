import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for customer churn prediction using state-of-the-art techniques.
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = None
        self.kmeans = None
        
    def create_temporal_features(self, df):
        """Create time-based features that capture customer behavior patterns."""
        print("Creating temporal features...")
        
        # Customer lifecycle stage
        df['LifecycleStage'] = pd.cut(df['remainder__tenure'], 
                                     bins=[0, 6, 12, 24, 60, 1000], 
                                     labels=['New', 'Early', 'Growth', 'Mature', 'Long-term'])
        
        # Tenure segments
        df['TenureSegment'] = pd.cut(df['remainder__tenure'], 
                                    bins=[0, 3, 6, 12, 24, 1000], 
                                    labels=['Very_New', 'New', 'Early', 'Established', 'Long_term'])
        
        # Monthly charge segments
        df['ChargeSegment'] = pd.cut(df['remainder__MonthlyCharges'], 
                                   bins=[0, 30, 50, 70, 100, 200], 
                                   labels=['Low', 'Medium', 'High', 'Premium', 'Ultra'])
        
        return df
    
    def create_interaction_features(self, df):
        """Create sophisticated interaction features between variables."""
        print("Creating interaction features...")
        
        # Service utilization score
        service_cols = [
            'remainder__OnlineSecurity', 'remainder__OnlineBackup', 
            'remainder__DeviceProtection', 'remainder__TechSupport',
            'remainder__StreamingTV', 'remainder__StreamingMovies'
        ]
        
        # Convert to numeric if needed
        for col in service_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
        
        df['ServiceUtilization'] = df[service_cols].sum(axis=1)
        
        # Value per service
        df['ValuePerService'] = df['remainder__MonthlyCharges'] / (df['ServiceUtilization'] + 1)
        
        # Contract efficiency
        df['ContractEfficiency'] = df['remainder__tenure'] * df['cat__Contract_Month-to-month']
        
        # Payment risk score
        payment_risk = 0
        payment_risk += (df['cat__PaymentMethod_Electronic check'] == 1) * 2
        payment_risk += (df['cat__PaymentMethod_Mailed check'] == 1) * 1
        df['PaymentRiskScore'] = payment_risk
        
        # Internet service risk
        internet_risk = 0
        internet_risk += (df['cat__InternetService_Fiber optic'] == 1) * 2
        internet_risk += (df['cat__InternetService_DSL'] == 1) * 1
        df['InternetRiskScore'] = internet_risk
        
        return df
    
    def create_behavioral_features(self, df):
        """Create features that capture customer behavior patterns."""
        print("Creating behavioral features...")
        
        # Customer value segments
        df['CustomerValue'] = df['remainder__MonthlyCharges'] * df['remainder__tenure']
        df['ValueSegment'] = pd.qcut(df['CustomerValue'], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
        
        # Service complexity
        df['ServiceComplexity'] = df['ServiceUtilization'] * df['remainder__MonthlyCharges']
        
        # Loyalty indicators
        df['LoyaltyScore'] = 0
        df.loc[df['remainder__tenure'] > 24, 'LoyaltyScore'] += 3
        df.loc[df['remainder__tenure'] > 12, 'LoyaltyScore'] += 2
        df.loc[df['remainder__tenure'] > 6, 'LoyaltyScore'] += 1
        df.loc[df['cat__Contract_Two year'] == 1, 'LoyaltyScore'] += 2
        df.loc[df['cat__Contract_One year'] == 1, 'LoyaltyScore'] += 1
        
        # Risk indicators
        df['RiskScore'] = 0
        df.loc[df['cat__Contract_Month-to-month'] == 1, 'RiskScore'] += 3
        df.loc[df['remainder__tenure'] < 6, 'RiskScore'] += 2
        df.loc[df['remainder__MonthlyCharges'] > 80, 'RiskScore'] += 1
        df.loc[df['cat__PaymentMethod_Electronic check'] == 1, 'RiskScore'] += 1
        
        return df
    
    def create_statistical_features(self, df):
        """Create statistical features using rolling windows and aggregations."""
        print("Creating statistical features...")
        
        # Z-score for monthly charges
        df['MonthlyChargesZScore'] = (df['remainder__MonthlyCharges'] - df['remainder__MonthlyCharges'].mean()) / df['remainder__MonthlyCharges'].std()
        
        # Percentile ranks
        df['TenurePercentile'] = df['remainder__tenure'].rank(pct=True)
        df['ChargesPercentile'] = df['remainder__MonthlyCharges'].rank(pct=True)
        
        # Ratio features
        df['TenureToChargeRatio'] = df['remainder__tenure'] / (df['remainder__MonthlyCharges'] + 1)
        df['ChargeToServiceRatio'] = df['remainder__MonthlyCharges'] / (df['ServiceUtilization'] + 1)
        
        return df
    
    def create_clustering_features(self, df):
        """Create features using unsupervised learning techniques."""
        print("Creating clustering features...")
        
        # Prepare features for clustering
        cluster_features = [
            'remainder__tenure', 'remainder__MonthlyCharges', 
            'ServiceUtilization', 'CustomerValue'
        ]
        
        # Scale features for clustering
        X_cluster = df[cluster_features].copy()
        X_cluster_scaled = self.scaler.fit_transform(X_cluster)
        
        # K-means clustering
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['CustomerSegment'] = self.kmeans.fit_predict(X_cluster_scaled)
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=3, random_state=42)
        pca_features = self.pca.fit_transform(X_cluster_scaled)
        df['PCA_Component_1'] = pca_features[:, 0]
        df['PCA_Component_2'] = pca_features[:, 1]
        df['PCA_Component_3'] = pca_features[:, 2]
        
        return df
    
    def create_advanced_risk_features(self, df):
        """Create sophisticated risk assessment features."""
        print("Creating advanced risk features...")
        
        # Churn probability estimate based on known factors
        df['EstimatedChurnProb'] = 0.0
        
        # Base probability
        base_prob = 0.26  # Industry average churn rate
        
        # Adjust based on contract type
        df.loc[df['cat__Contract_Month-to-month'] == 1, 'EstimatedChurnProb'] += 0.3
        df.loc[df['cat__Contract_One year'] == 1, 'EstimatedChurnProb'] += 0.1
        df.loc[df['cat__Contract_Two year'] == 1, 'EstimatedChurnProb'] -= 0.2
        
        # Adjust based on tenure
        df.loc[df['remainder__tenure'] < 6, 'EstimatedChurnProb'] += 0.2
        df.loc[df['remainder__tenure'] < 12, 'EstimatedChurnProb'] += 0.1
        df.loc[df['remainder__tenure'] > 24, 'EstimatedChurnProb'] -= 0.15
        
        # Adjust based on payment method
        df.loc[df['cat__PaymentMethod_Electronic check'] == 1, 'EstimatedChurnProb'] += 0.1
        
        # Adjust based on internet service
        df.loc[df['cat__InternetService_Fiber optic'] == 1, 'EstimatedChurnProb'] += 0.05
        
        # Cap probability
        df['EstimatedChurnProb'] = df['EstimatedChurnProb'].clip(0, 1)
        
        # Risk tier classification
        df['RiskTier'] = pd.cut(df['EstimatedChurnProb'], 
                               bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                               labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
        
        return df
    
    def engineer_all_features(self, df):
        """Apply all advanced feature engineering techniques."""
        print("Starting advanced feature engineering...")
        
        # Apply all feature engineering steps
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        df = self.create_behavioral_features(df)
        df = self.create_statistical_features(df)
        df = self.create_clustering_features(df)
        df = self.create_advanced_risk_features(df)
        
        print(f"Advanced feature engineering completed. Total features: {df.shape[1]}")
        return df

def engineer_advanced_features():
    """Main function to run advanced feature engineering."""
    # Read processed data
    df = pd.read_csv('data/processed_data.csv')
    
    # Handle TotalCharges
    df['remainder__TotalCharges'] = pd.to_numeric(
        df['remainder__TotalCharges'].replace(' ', np.nan),
        errors='coerce'
    )
    
    # Initialize advanced feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Apply advanced feature engineering
    df = engineer.engineer_all_features(df)
    
    # Save engineered data
    df.to_csv('data/advanced_engineered_data.csv', index=False)
    print("Advanced feature engineering completed and saved to data/advanced_engineered_data.csv")
    
    return df

if __name__ == "__main__":
    engineer_advanced_features()