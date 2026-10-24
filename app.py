from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

st.set_page_config(page_title="Mileage Predictor", page_icon="🚗")

DATA = Path(__file__).parent / "81_dataset.csv"
FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "cyl": d["cylinders"],
        "lndisp": np.log(d["displacement"]),
        "lnhp": np.log(d["horsepower"]),
        "lnw": np.log(d["weight"]),
        "acc": d["acceleration"],
        "yr": d["model year"],
        "o2": (d["origin"] == 2).astype(int),
        "o3": (d["origin"] == 3).astype(int),
    })


@st.cache_resource(show_spinner="Training the model on the dataset...")
def load():
    df = pd.read_csv(DATA, na_values="?")
    median_hp = float(df["horsepower"].median())
    df["horsepower"] = df["horsepower"].fillna(median_hp)
    X, y = build_features(df), np.log(df["mpg"])
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=0)
    cv = np.exp(cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0)))
    err = df["mpg"] - cv
    meta = {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "r2": float(1 - (err ** 2).sum() / ((df["mpg"] - df["mpg"].mean()) ** 2).sum()),
        "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES[:6]},
    }
    return model.fit(X, y), meta, df


model, meta, df = load()

st.title("Mileage Predictor")
st.caption("Trained on 398 cars from model years 1970 to 1982.")

c1, c2 = st.columns(2)
cyl = c1.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
year = c2.slider("Model year", 1970, 1982, 1976) - 1900
origin = st.radio("Built in", [1, 2, 3], format_func={1: "USA", 2: "Europe", 3: "Japan"}.get, horizontal=True)
disp = st.number_input("Displacement (cubic inches)", 30.0, 600.0, 225.0, 1.0)
hp = st.number_input("Horsepower", 20.0, 400.0, 100.0, 1.0)
wt = st.number_input("Weight (lb)", 1000.0, 6500.0, 3300.0, 10.0)
acc = st.number_input("0 to 60 mph (seconds)", 6.0, 30.0, 16.0, 0.1)

row = pd.DataFrame([{"cylinders": cyl, "displacement": disp, "horsepower": hp, "weight": wt,
                     "acceleration": acc, "model year": year, "origin": origin}])
pred = float(np.exp(model.predict(build_features(row))[0]))

st.metric("Estimated mileage", f"{pred:.1f} mpg")
st.write(f"Likely range: {max(pred - meta['rmse'], 1):.0f} to {pred + meta['rmse']:.0f} mpg. "
         f"That is {pred * 0.425144:.1f} km/l, or {235.215 / pred:.1f} l/100 km.")

for name, val, label in [("displacement", disp, "Displacement"), ("horsepower", hp, "Horsepower"),
                         ("weight", wt, "Weight"), ("acceleration", acc, "Acceleration")]:
    lo, hi = meta["ranges"][name]
    if not lo <= val <= hi:
        st.warning(f"{label} is outside the training data ({lo:.0f} to {hi:.0f}), so treat this estimate with caution.")

with st.expander("Closest cars in the dataset"):
    sd = {"displacement": 104, "horsepower": 38, "weight": 850, "cylinders": 1.7}
    d = sum(((df[k] - row[k][0]) / s) ** 2 for k, s in sd.items()) + 0.5 * ((df["model year"] - year) / 3.7) ** 2
    st.dataframe(df.loc[d.nsmallest(5).index, ["car name", "model year", "cylinders", "horsepower", "weight", "mpg"]],
                 hide_index=True)

st.caption(f"Cross-validated: R² {meta['r2']:.2f}, typical error {meta['mae']:.1f} mpg.")
