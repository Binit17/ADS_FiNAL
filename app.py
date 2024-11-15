import streamlit as st
from bertopic import BERTopic
import pandas as pd

# Load the trained BERTopic model
@st.cache_resource
def load_model():
    return BERTopic.load("Trained_models/bertopic_news_model")

# Load the dataset
@st.cache_resource
def load_dataset():
    import json
    with open("samplesentenceList.json", "r") as file:
        return json.load(file)

# Load model and dataset
topic_model = load_model()
docs = load_dataset()

# Visualizations
def visualize_barchart():
    st.subheader("Barchart of Top Words per Topic")
    fig = topic_model.visualize_barchart()
    st.plotly_chart(fig)

def visualize_heatmap():
    st.subheader("Topic-Topic Similarity Heatmap")
    fig = topic_model.visualize_heatmap()
    st.plotly_chart(fig)

def visualize_hierarchy():
    st.subheader("Topic Hierarchy (Dendrogram)")
    fig = topic_model.visualize_hierarchy()
    st.plotly_chart(fig)

def visualize_topics():
    st.subheader("Topic Overview")
    fig = topic_model.visualize_topics()
    st.plotly_chart(fig)

# Main app
def main():
    st.title("BERTopic Visualization Dashboard")

    st.sidebar.title("Navigation")
    visualizations = [
        "Barchart of Top Words per Topic",
        "Topic-Topic Heatmap",
        "Topic Hierarchy",
        "Topic Overview",
    ]
    selected_vis = st.sidebar.selectbox("Select Visualization", visualizations)

    if selected_vis == "Barchart of Top Words per Topic":
        visualize_barchart()

    elif selected_vis == "Topic-Topic Heatmap":
        visualize_heatmap()

    elif selected_vis == "Topic Hierarchy":
        visualize_hierarchy()

    elif selected_vis == "Topic Overview":
        visualize_topics()

# Run the app
if __name__ == "__main__":
    main()
