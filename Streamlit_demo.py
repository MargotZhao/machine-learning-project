import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("📊 Upsell & Customer Segmentation Dashboard")

# ============================
# SECTION 1: Upsell Data Upload
# ============================
st.header("📈 Upsell Behavior Analysis")

upsell_file = st.file_uploader("Upload 'upsell_cleaned.csv'", type=["csv"])

if upsell_file is not None:
    df = pd.read_csv(upsell_file)
    st.write("Columns:", df.columns.tolist())  # Show columns to debug

    # Flexible datetime column lookup
    date_col = next((col for col in ['Create date', 'Create Date', 'create_date'] if col in df.columns), None)
    close_col = next((col for col in ['Close date', 'Close Date', 'close_date'] if col in df.columns), None)

    if date_col and close_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[close_col] = pd.to_datetime(df[close_col], errors='coerce')
        df['Did_Upsell'] = df['Did_Upsell'].astype(int)

        st.subheader("Summary Table")
        summary = df.groupby('Did_Upsell').agg({
            'Technology_Count': 'mean',
            'Kickoff Call': lambda x: x.notna().mean(),
            'Create_Quarter': 'mean'
        }).T
        st.dataframe(summary)

        st.subheader("Visualizations")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Technology Count by Upsell**")
            fig1, ax1 = plt.subplots()
            sns.barplot(x='Did_Upsell', y='Technology_Count', data=df, ax=ax1)
            ax1.set_xticklabels(['No Upsell', 'Upsell'])
            st.pyplot(fig1)

        with col2:
            st.markdown("**Kickoff Call Presence Rate**")
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Did_Upsell', y=df['Kickoff Call'].notna().astype(int), data=df, ax=ax2)
            ax2.set_xticklabels(['No Upsell', 'Upsell'])
            st.pyplot(fig2)

        st.markdown("**Create Quarter by Upsell**")
        fig3, ax3 = plt.subplots()
        sns.barplot(x='Did_Upsell', y='Create_Quarter', data=df, ax=ax3)
        ax3.set_xticklabels(['No Upsell', 'Upsell'])
        st.pyplot(fig3)
    else:
        st.error("❌ Could not find 'Create date' or 'Close date' columns.")


# ===============================
# SECTION 2: Customer Segmentation
# ===============================
st.header("🧠 Customer Segmentation via KMeans")

cluster_file = st.file_uploader("Upload 'merged_cleaned.csv'", type=["csv"], key='clustering')

if cluster_file is not None:
    merged_cleaned = pd.read_csv(cluster_file)

    if 'Last Activity Date' in merged_cleaned.columns:
        merged_cleaned['Last Activity Date'] = pd.to_datetime(merged_cleaned['Last Activity Date'], errors='coerce')
        today = pd.Timestamp.today()
        merged_cleaned['Recency_Days'] = (today - merged_cleaned['Last Activity Date']).dt.days

        customer_df = merged_cleaned.groupby('Associated Company (Primary)').agg({
            'Amount': ['sum', 'mean', 'count'],
            'Deal probability': 'mean',
            'Deal Score': 'mean',
            'Is Closed Won': 'sum',
            'Is closed lost': 'sum',
            'Is Open (numeric)': 'sum',
            'Days to close': 'mean',
            'Recency_Days': 'min'
        })

        customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
        customer_df.reset_index(inplace=True)
        customer_df.dropna(inplace=True)

        features = customer_df.drop(columns=['Associated Company (Primary)'])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=4, random_state=42)
        customer_df['Cluster'] = kmeans.fit_predict(scaled_features)

        cluster_summary = customer_df.groupby('Cluster').mean(numeric_only=True)

        st.subheader("Cluster Summary Table")
        st.dataframe(cluster_summary)

        st.subheader("Cluster Visualizations")

        fig4, ax4 = plt.subplots()
        sns.countplot(data=customer_df, x='Cluster', palette='Set2', ax=ax4)
        ax4.set_title("Number of Customers per Cluster")
        st.pyplot(fig4)

        fig5, ax5 = plt.subplots()
        sns.barplot(data=cluster_summary.reset_index(), x='Cluster', y='Amount_mean', palette='Set1', ax=ax5)
        ax5.set_title("Average Deal Amount by Cluster")
        st.pyplot(fig5)

        fig6, ax6 = plt.subplots()
        sns.barplot(data=cluster_summary.reset_index(), x='Cluster', y='Deal Score_mean', palette='Set3', ax=ax6)
        ax6.set_title("Average Deal Score by Cluster")
        st.pyplot(fig6)

        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.heatmap(cluster_summary, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax7)
        ax7.set_title("Cluster Feature Heatmap")
        st.pyplot(fig7)
    else:
        st.error("❌ 'Last Activity Date' column not found.")
