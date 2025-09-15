import os, glob, warnings
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ถ้ามี xgboost
try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")


# ----------------------------
# Enhanced Image Features
# ----------------------------
def enhanced_image_features(img_path):
    """Extract comprehensive image features including geometric and texture features"""
    try:
        img = Image.open(img_path).convert("RGB")
        # Keep original aspect ratio for better geometric features
        img_resized = img.resize((224, 224))

        # Convert to different color spaces
        arr = np.asarray(img_resized).astype(np.float32) / 255.0
        img_cv = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Basic color features
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])

        # HSV color space
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Geometric features (estimate fruit size from image)
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Find contours for size estimation
        _, binary = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour (assumed to be the fruit)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            # Bounding box features
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 1
            extent = area / (w * h) if (w * h) > 0 else 0

            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
        else:
            area = perimeter = aspect_ratio = extent = circularity = solidity = 0

        # Texture features (simple)
        gray_int = (gray * 255).astype(np.uint8)

        # Calculate gradients for texture
        grad_x = cv2.Sobel(gray_int, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_int, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        features = {
            # Basic color statistics
            "r_mean": r.mean(), "g_mean": g.mean(), "b_mean": b.mean(),
            "r_std": r.std(), "g_std": g.std(), "b_std": b.std(),
            "bright_mean": gray.mean(), "bright_std": gray.std(),

            # HSV features
            "h_mean": h.mean() / 255.0, "s_mean": s.mean() / 255.0, "v_mean": v.mean() / 255.0,
            "h_std": h.std() / 255.0, "s_std": s.std() / 255.0, "v_std": v.std() / 255.0,

            # Geometric features (normalized by image size)
            "area_ratio": area / (224 * 224),
            "perimeter_ratio": perimeter / (4 * 224),
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "circularity": circularity,
            "solidity": solidity,

            # Texture features
            "gradient_mean": gradient_magnitude.mean() / 255.0,
            "gradient_std": gradient_magnitude.std() / 255.0,

            # Color distribution features
            "r_q25": np.percentile(r, 25), "r_q75": np.percentile(r, 75),
            "g_q25": np.percentile(g, 25), "g_q75": np.percentile(g, 75),
            "b_q25": np.percentile(b, 25), "b_q75": np.percentile(b, 75),
        }

        return features

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return {k: 0 for k in ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
                               "bright_mean", "bright_std", "h_mean", "s_mean", "v_mean",
                               "h_std", "s_std", "v_std", "area_ratio", "perimeter_ratio",
                               "aspect_ratio", "extent", "circularity", "solidity",
                               "gradient_mean", "gradient_std", "r_q25", "r_q75",
                               "g_q25", "g_q75", "b_q25", "b_q75"]}


# ----------------------------
# Feature Engineering
# ----------------------------
def create_engineered_features(data):
    """Create additional engineered features"""
    df = data.copy()

    # Physical relationships
    df['weight_density_est'] = df['weight_g'] / (df['area_ratio'] * 1000)  # rough density estimate
    df['weight_sqrt'] = np.sqrt(df['weight_g'])
    df['weight_squared'] = df['weight_g'] ** 2
    df['weight_log'] = np.log1p(df['weight_g'])

    # Color ratios and combinations
    df['rg_ratio'] = df['r_mean'] / (df['g_mean'] + 1e-6)
    df['rb_ratio'] = df['r_mean'] / (df['b_mean'] + 1e-6)
    df['gb_ratio'] = df['g_mean'] / (df['b_mean'] + 1e-6)
    df['color_variance'] = (df['r_std'] + df['g_std'] + df['b_std']) / 3
    df['brightness_contrast'] = df['bright_std'] / (df['bright_mean'] + 1e-6)

    # Shape complexity
    df['shape_complexity'] = df['perimeter_ratio'] / (df['area_ratio'] + 1e-6)
    df['roundness'] = df['circularity'] * df['solidity']

    return df


# ----------------------------
# Metric helper
# ----------------------------
def report(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    print(f"{prefix}MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f} | MAPE={mape:.1f}%")
    return mae, rmse, r2, mape


# ----------------------------
# Cross-validation helper
# ----------------------------
def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation and return mean scores"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))

    return {
        'r2_mean': scores.mean(),
        'r2_std': scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std()
    }


# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading data...")
    # โหลด CSV ของคุณเอง
    df = pd.read_csv("./data/mangosteen-train-3.csv")

    # สมมุติ: มีคอลัมน์ "ID","Weight (g)","ปริมาตร"
    df = df.rename(columns={"ID": "id", "Weight (g)": "weight_g", "ปริมาตร": "volumes_ml"})
    df["id"] = df["id"].astype(str).apply(lambda x: f"i{int(x):03d}" if x.isnumeric() else x)

    print("Extracting enhanced image features...")
    rows = []
    for _, r in df.iterrows():
        img_path = glob.glob(f"./data/prepared-224-112/{r['id']}*.jpg")
        if not img_path:
            continue

        feats = enhanced_image_features(img_path[0])
        rows.append({
            "weight_g": r["weight_g"],
            "volume_ml": r["volumes_ml"],
            **feats
        })

    data = pd.DataFrame(rows)
    print(f"Samples: {len(data)}")

    # Feature engineering
    print("Creating engineered features...")
    data = create_engineered_features(data)

    # Remove outliers (optional)
    Q1 = data['volume_ml'].quantile(0.25)
    Q3 = data['volume_ml'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data['volume_ml'] >= Q1 - 1.5 * IQR) & (data['volume_ml'] <= Q3 + 1.5 * IQR)]
    print(f"Samples after outlier removal: {len(data)}")

    X = data.drop(columns=["volume_ml"]).values
    y = data["volume_ml"].values

    # Feature importance analysis (using Random Forest)
    print("\nAnalyzing feature importance...")
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    feature_names = data.drop(columns=["volume_ml"]).columns
    importances = rf_temp.feature_importances_

    # Sort features by importance
    importance_idx = np.argsort(importances)[::-1]
    print("Top 10 most important features:")
    for i in range(min(10, len(importance_idx))):
        idx = importance_idx[i]
        print(f"{feature_names[idx]:20s}: {importances[idx]:.4f}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    # ---------------- Enhanced Models ----------------
    models = {
        "Linear": Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
        "Polynomial": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", LinearRegression())
        ]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("reg", RidgeCV(alphas=np.logspace(-4, 4, 20)))]),
        "Lasso": Pipeline(
            [("scaler", StandardScaler()), ("reg", LassoCV(alphas=np.logspace(-4, 1, 20), max_iter=10000))]),
        "ElasticNet": Pipeline(
            [("scaler", StandardScaler()), ("reg", ElasticNetCV(l1_ratio=[.1, .5, .7, .9], max_iter=10000))]),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=5, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                      random_state=42),
        "SVR": Pipeline([("scaler", StandardScaler()), ("reg", SVR(kernel="rbf", C=100, epsilon=0.1, gamma='scale'))]),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("reg", KNeighborsRegressor(n_neighbors=7, weights='distance'))]),
    }

    if XGB_AVAILABLE:
        models["XGB"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    # ---------------- Train & Evaluate ----------------
    print("\n" + "=" * 60)
    print("Training and evaluating models...")
    print("=" * 60)

    results = {}
    cv_results = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")

        # Cross-validation
        cv_scores = cross_validate_model(model, X_tr, y_tr)
        cv_results[name] = cv_scores
        print(f"CV R² = {cv_scores['r2_mean']:.4f} (±{cv_scores['r2_std']:.4f})")

        # Train and test
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mae, rmse, r2, mape = report(y_te, y_pred, prefix=f"{name} Test | ")
        results[name] = (mae, rmse, r2, mape)

    # Create ensemble model
    print("\n=== Ensemble Model ===")
    ensemble_models = []
    if XGB_AVAILABLE:
        ensemble_models = [
            ('rf', models['RandomForest']),
            ('xgb', models['XGB']),
            ('gb', models['GradientBoosting'])
        ]
    else:
        ensemble_models = [
            ('rf', models['RandomForest']),
            ('gb', models['GradientBoosting']),
            ('svr', models['SVR'])
        ]

    ensemble = VotingRegressor(ensemble_models)
    ensemble.fit(X_tr, y_tr)
    y_pred_ensemble = ensemble.predict(X_te)
    mae, rmse, r2, mape = report(y_te, y_pred_ensemble, prefix="Ensemble Test | ")
    results['Ensemble'] = (mae, rmse, r2, mape)

    # Summary tables
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8} | {'MAPE':<8} | {'CV R²':<8}")
    print("-" * 80)

    for name, (mae, rmse, r2, mape) in results.items():
        cv_r2 = cv_results.get(name, {}).get('r2_mean', 0)
        print(f"{name:<15} | {mae:<8.2f} | {rmse:<8.2f} | {r2:<8.4f} | {mape:<8.1f} | {cv_r2:<8.4f}")

    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k][2])  # Best R²
    print(f"\nBest model: {best_model} (R² = {results[best_model][2]:.4f})")

    # Feature importance for best tree-based model
    if best_model in ['RandomForest', 'XGB', 'GradientBoosting', 'Ensemble']:
        print(f"\nFeature importance analysis for {best_model}:")
        if best_model == 'Ensemble':
            # Use Random Forest from ensemble
            best_estimator = ensemble.named_estimators_['rf']
        else:
            best_estimator = models[best_model]

        if hasattr(best_estimator, 'feature_importances_'):
            importances = best_estimator.feature_importances_
        elif hasattr(best_estimator, 'named_steps') and hasattr(best_estimator.named_steps['reg'],
                                                                'feature_importances_'):
            importances = best_estimator.named_steps['reg'].feature_importances_
        else:
            importances = None

        if importances is not None:
            importance_idx = np.argsort(importances)[::-1]
            print("Top 10 features:")
            for i in range(min(10, len(importance_idx))):
                idx = importance_idx[i]
                print(f"  {feature_names[idx]:25s}: {importances[idx]:.4f}")


if __name__ == "__main__":
    main()