try:
    import joblib  # type: ignore[import-not-found]
except ModuleNotFoundError:
    joblib = None

try:
    import pandas as pd  # type: ignore[import-not-found]
except ModuleNotFoundError:
    pd = None

try:
    import numpy as np  # type: ignore[import-not-found]
except ModuleNotFoundError:
    np = None

try:
    import streamlit as st  # type: ignore[import-not-found]
except ModuleNotFoundError:
    st = None

if pd is None or np is None or st is None:
    raise ModuleNotFoundError(
        "Required packages are missing. Install pandas, numpy, and streamlit before running this app."
    )

# Page Configuration
st.set_page_config(
    page_title="Shopper Spectrum Hub",
    page_icon="🛍️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #28a745;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .cluster-card {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        border: 2px solid #007bff;
    }
    </style>
""", unsafe_allow_html=True)

# Safely load payload
@st.cache_resource
def load_data():
    return joblib.load('rfm_model1.pkl')

try:
    payload = load_data()
except Exception as e:
    st.error(f"Failed to load `rfm_model1.pkl`. Please make sure the file is in the same folder as `app.py`. Error details: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.title("🛍️ Shopper Spectrum")
module = st.sidebar.radio(
    "Select Module", 
    ["1️⃣ Product Recommendation", "2️⃣ Customer Segmentation"]
)

# ----------------------------------------------------
# 1️⃣ PRODUCT RECOMMENDATION MODULE
# ----------------------------------------------------
if module == "1️⃣ Product Recommendation":
    st.title("🎯 Product Recommendation Module")
    st.write("Get 5 complementary product recommendations based on item collaborative filtering.")
    st.divider()

    if 'item_similarity_df' not in payload or 'product_list' not in payload:
        st.error("Recommendation data not found in `rfm_model.pkl`. Re-run Step 1 in Colab to update the pickle file.")
        st.stop()

    similarity_df = payload['item_similarity_df']
    product_list = payload['product_list']

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_product = st.selectbox("Type or select a Product Name:", product_list)

    with col2:
        st.write("##")
        get_rec = st.button("Get Recommendations", use_container_width=True)

    if get_rec and selected_product:
        sim_scores = similarity_df[selected_product].sort_values(ascending=False)
        top_5 = sim_scores.iloc[1:6]

        st.subheader(f"Top 5 Recommendations for: {selected_product}")
        for idx, (prod_name, score) in enumerate(top_5.items(), start=1):
            st.markdown(
                f"""
                <div class="metric-card">
                    <b>#{idx} {prod_name}</b><br>
                    <small>Similarity Match: {score:.3f}</small>
                </div>
                """, 
                unsafe_allow_html=True
            )

# ----------------------------------------------------
# 2️⃣ CUSTOMER SEGMENTATION MODULE
# ----------------------------------------------------
elif module == "2️⃣ Customer Segmentation":
    st.title("🎯 Customer Segmentation Module")
    st.write("Predict the customer segment using RFM metrics.")
    st.divider()

    model = payload['model']
    scaler = payload['scaler']
    features = payload.get('features', ['Recency', 'Frequency', 'Monetary'])
    cluster_labels = payload.get('cluster_labels', {})

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Input RFM Values")
        recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)
        frequency = st.number_input("Frequency (Total number of purchases)", min_value=1, value=5)
        monetary = st.number_input("Monetary (Total spend in $)", min_value=0.0, value=500.0)

        predict_btn = st.button("Predict Cluster", use_container_width=True)

    with col_out:
        st.subheader("Segmentation Output")
        if predict_btn:
            input_df = pd.DataFrame([[recency, frequency, monetary]], columns=features)
            scaled_data = scaler.transform(input_df)
            cluster_id = int(model.predict(scaled_data)[0])
            segment_name = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")

            st.markdown(
                f"""
                <div class="cluster-card">
                    <h3>Assigned Segment</h3>
                    <h2>{segment_name}</h2>
                    <p>Cluster ID: {cluster_id}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )