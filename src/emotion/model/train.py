from src.emotion.model.regressor import EmotionRegressor
from src.emotion.model.util import get_features, get_valence_targets, get_arousal_targets, FEATURES_DIR, SELECTOR_EXT
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


def train(
    valence_regressor, arousal_regressor,
    valence_features_train, arousal_features_train,
    valence_targets_train, arousal_targets_train
) -> None:
    valence_regressor.fit(valence_features_train, valence_targets_train)
    arousal_regressor.fit(arousal_features_train, arousal_targets_train)


def test(
    valence_regressor, arousal_regressor,
    valence_features_test, arousal_features_test,
    valence_targets_test, arousal_targets_test
) -> None:
    valence_targets_predict = valence_regressor.predict(valence_features_test)
    arousal_targets_predict = arousal_regressor.predict(arousal_features_test)

    valence_mse = mean_squared_error(valence_targets_test, valence_targets_predict)
    arousal_mse = mean_squared_error(arousal_targets_test, arousal_targets_predict)
    valence_mae = mean_absolute_error(valence_targets_test, valence_targets_predict)
    arousal_mae = mean_absolute_error(arousal_targets_test, arousal_targets_predict)
    print(f"MSE: (valence: {valence_mse:.4f}, arousal: {arousal_mse:.4f})")
    print(f"MAE: (valence: {valence_mae:.4f}, arousal: {arousal_mae:.4f})")


def train_and_test() -> None:
    valence_regressor, arousal_regressor = EmotionRegressor(), EmotionRegressor()

    opensmile_features = get_features("opensmile")

    opensmile_valence_selector = joblib.load(f"{FEATURES_DIR}/opensmile_valence.{SELECTOR_EXT}")
    opensmile_arousal_selector = joblib.load(f"{FEATURES_DIR}/opensmile_arousal.{SELECTOR_EXT}")

    valence_features = opensmile_valence_selector.transform(opensmile_features)
    arousal_features = opensmile_arousal_selector.transform(opensmile_features)

    valence_targets = get_valence_targets()
    arousal_targets = get_arousal_targets()

    valence_features_train, valence_features_test, valence_targets_train, valence_targets_test = train_test_split(
        valence_features, valence_targets, test_size=0.2, random_state=42
    )
    arousal_features_train, arousal_features_test, arousal_targets_train, arousal_targets_test = train_test_split(
        arousal_features, arousal_targets, test_size=0.2, random_state=42
    )

    train(
        valence_regressor, arousal_regressor,
        valence_features_train, arousal_features_train,
        valence_targets_train, arousal_targets_train
    )
    test(
        valence_regressor, arousal_regressor,
        valence_features_test, arousal_features_test,
        valence_targets_test, arousal_targets_test
    )


if __name__ == "__main__":
    train_and_test()
