# ─── Imports ───
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ─── Page config (Task 1) ───
st.set_page_config(page_title="Housing Predictor", page_icon="🏠", layout="wide")

# ─── Cached functions (Tasks 1 & 6) ───
@st.cache_data
def load_data():
    df = pd.read_csv("data/housing_madrid.csv")
    return df

@st.cache_resource
def train_model(df):
    feature_cols = ["area_sqm", "bedrooms", "bathrooms", "year_built",
                    "neighborhood", "property_type", "condition", "energy_rating", "has_parking"]
    X = df[feature_cols].copy()
    y = df["price_eur"]
    X = pd.get_dummies(X, columns=["neighborhood", "property_type", "condition", "energy_rating"],
                       drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    return model, r2, X.columns.tolist()

# ─── Load data and train model ───
df = load_data()
model, r2, model_columns = train_model(df)

# ─── Title and navigation (Task 1) ───
st.title("🏠 Madrid Housing Price Predictor")
page = st.sidebar.radio("Navigate", ["📊 Data Explorer", "📈 Visualizations", "🤖 ML Predictor"])

# ═══════════════════════════════════════════════════════════════
# PAGE 1: DATA EXPLORER (Task 2)
# ═══════════════════════════════════════════════════════════════
if page == "📊 Data Explorer":
    st.header("📊 Data Explorer")

    # Sidebar filters
    selected_neighborhoods = st.sidebar.multiselect(
        "Neighborhoods",
        df["neighborhood"].unique().tolist(),
        default=df["neighborhood"].unique().tolist()
    )
    selected_types = st.sidebar.multiselect(
        "Property Types",
        df["property_type"].unique().tolist(),
        default=df["property_type"].unique().tolist()
    )

    # Filter the data
    filtered_df = df[
        df["neighborhood"].isin(selected_neighborhoods) &
        df["property_type"].isin(selected_types)
    ]

    # KPI metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Properties", f"{len(filtered_df)}")
    with col2:
        st.metric("Median Price", f"€{filtered_df['price_eur'].median():,.0f}")
    with col3:
        st.metric("Avg Area", f"{filtered_df['area_sqm'].mean():.0f} m²")

    # Data table
    st.dataframe(filtered_df, use_container_width=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe())

# ═══════════════════════════════════════════════════════════════
# PAGE 2: VISUALIZATIONS (Tasks 3, 4, 5)
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.header("📈 Visualizations")

    # ─── Task 3: Scatter plot ───
    st.subheader("Price vs Area")

    color_by = st.selectbox(
        "Color by",
        ["neighborhood", "property_type", "condition", "energy_rating"]
    )

    fig = px.scatter(
        df,
        x="area_sqm",
        y="price_eur",
        color=color_by,
        hover_data=["bedrooms", "neighborhood", "condition"],
        labels={"area_sqm": "Area (m²)", "price_eur": "Price (€)"},
        title="Price vs Area"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Task 4: Bar chart and box plot ───
    st.subheader("Price by Neighborhood")

    median_prices = df.groupby("neighborhood")["price_eur"].median().reset_index()
    median_prices = median_prices.sort_values("price_eur", ascending=True)

    fig = px.bar(
        median_prices, x="price_eur", y="neighborhood", orientation="h",
        title="Median Price by Neighborhood",
        labels={"price_eur": "Median Price (€)", "neighborhood": ""}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Distribution")

    fig = px.box(
        df, x="neighborhood", y="price_eur", color="neighborhood",
        title="Price Distribution by Neighborhood",
        labels={"price_eur": "Price (€)", "neighborhood": ""}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Task 5: Correlation heatmap ───
    st.subheader("Correlation Heatmap")

    numeric_cols = ["area_sqm", "bedrooms", "bathrooms", "year_built", "price_eur", "price_per_sqm"]
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Values close to +1 or -1 indicate strong correlation. Values near 0 indicate weak correlation.")

# ═══════════════════════════════════════════════════════════════
# PAGE 3: ML PREDICTOR (Tasks 6, 7)
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 ML Predictor":
    st.header("🤖 ML Predictor")

    # ─── Task 6: Model info ───
    st.metric("Model R² Score", f"{r2:.3f}")

    if r2 > 0.90:
        st.success("Excellent model performance!")
    elif r2 > 0.75:
        st.info("Good model performance.")
    else:
        st.warning("Model needs improvement.")

    st.subheader("Feature Importances")

    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True).tail(10)

    fig = px.bar(
        importance_df, x="Importance", y="Feature", orientation="h",
        title="Top 10 Most Important Features"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Task 7: Interactive predictor ───
    st.subheader("Predict a Property's Price")
    st.markdown("Adjust the inputs below to describe a property and see the predicted price.")

    left, right = st.columns(2)

    with left:
        area = st.slider("Area (m²)", 25, 300, 100)
        bedrooms = st.slider("Bedrooms", 1, 6, 2)
        bathrooms = st.slider("Bathrooms", 1, 4, 1)
        year_built = st.slider("Year Built", 1960, 2024, 2000)

    with right:
        neighborhood = st.selectbox("Neighborhood", df["neighborhood"].unique())
        property_type = st.selectbox("Property Type", df["property_type"].unique())
        condition = st.selectbox("Condition", df["condition"].unique())
        energy_rating = st.selectbox("Energy Rating", sorted(df["energy_rating"].unique()))
        has_parking = st.checkbox("Has Parking", value=True)

    # Build input and predict
    input_dict = {
        "area_sqm": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "has_parking": has_parking,
        "neighborhood": neighborhood,
        "property_type": property_type,
        "condition": condition,
        "energy_rating": energy_rating
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(
        input_df,
        columns=["neighborhood", "property_type", "condition", "energy_rating"],
        drop_first=True
    )
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    predicted_price = model.predict(input_encoded)[0]

    st.success(f"Estimated Price: **€{predicted_price:,.0f}**")
    st.metric("Price per m²", f"€{predicted_price / area:,.0f}/m²")
