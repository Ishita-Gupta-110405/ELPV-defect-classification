import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from preprocessing import get_splits

def extract_hog_features(img_array):
    print("Extracting HOG features. This takes a moment...")
    features = [hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2)) for img in img_array]
    return np.array(features)

def main():
    X_train, X_val, _, y_train, y_val, _ = get_splits()
    
    X_train_hog = extract_hog_features(X_train)
    X_val_hog = extract_hog_features(X_val)
    
    print("Training HOG + SVM baseline...")
    svm_clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm_clf.fit(X_train_hog, y_train)
    
    print("Evaluating on Validation Set...")
    val_preds = svm_clf.predict(X_val_hog)
    acc = accuracy_score(y_val, val_preds)
    print(f"\n--- HOG+SVM Preliminary Validation Accuracy: {acc * 100:.2f}% ---")

if __name__ == "__main__":
    main()