import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def classify_security_events(features, labels):
    """
    Supervised classification for security event monitoring.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    return clf, score

if __name__ == "__main__":
    print("Security classification module initialized.")