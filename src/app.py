from utils import db_connect, load_processed_data
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

def main():
    
    engine = db_connect()
    
    
    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    
    train_df = load_processed_data(train_path)
    test_df = load_processed_data(test_path)
    
    
    X_train = train_df.drop('Outcome', axis=1)
    y_train = train_df['Outcome']
    X_test = test_df.drop('Outcome', axis=1)
    y_test = test_df['Outcome']
    
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    
    param_grid = {
        "Decision Tree": {"max_depth": [3, 5, 10], "min_samples_split": [2, 10]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.1, 0.05], "max_depth": [3, 5]}
    }
    
   
    for model_name, model in models.items():
        print(f"\nEvaluando {model_name}...")
        
        
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Mejores parámetros para {model_name}: {grid_search.best_params_}")
        print(f"Accuracy {model_name}: {accuracy:.4f}")
        print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
        
       
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{model_name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"Modelo {model_name} guardado en: {model_path}")
        
      
        if model_name == "Gradient Boosting":
            importances = best_model.feature_importances_
            features = X_train.columns
            sns.barplot(x=importances, y=features)
            plt.title(f"Importancia de características - {model_name}")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
