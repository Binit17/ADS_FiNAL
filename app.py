import streamlit as st
from bertopic import BERTopic

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

# Front Page with All Visualizations
def front_page():
    st.title("BERTopic Visualizations Dashboard")
    st.write("Explore the various visualizations provided by BERTopic on this consolidated page!")

    # Barchart
    st.subheader("1. Barchart of Top Words per Topic")
    fig_barchart = topic_model.visualize_barchart()
    st.plotly_chart(fig_barchart, use_container_width=True)

    # Heatmap
    st.subheader("2. Topic-Topic Similarity Heatmap")
    fig_heatmap = topic_model.visualize_heatmap()
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Hierarchy
    st.subheader("3. Topic Hierarchy (Dendrogram)")
    fig_hierarchy = topic_model.visualize_hierarchy()
    st.plotly_chart(fig_hierarchy, use_container_width=True)

    # Topic Overview
    st.subheader("4. Topic Overview")
    fig_topics = topic_model.visualize_topics()
    st.plotly_chart(fig_topics, use_container_width=True)

# Individual Visualizations
def visualize_barchart():
    st.subheader("Barchart of Top Words per Topic")
    
    # Input: Number of topics to display
    num_topics = st.slider(
        "Select the number of topics to visualize",
        min_value=1,
        max_value=len(topic_model.get_topics()),
        value=5,
        step=1,
    )
    
    # Generate the barchart
    fig = topic_model.visualize_barchart(top_n_topics=num_topics)
    st.plotly_chart(fig, use_container_width=True)

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

# Main App
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Front Page (All Visualizations)": front_page,
        "Barchart of Top Words per Topic": visualize_barchart,
        "Topic-Topic Heatmap": visualize_heatmap,
        "Topic Hierarchy": visualize_hierarchy,
        "Topic Overview": visualize_topics,
    }

    # Sidebar navigation
    selected_page = st.sidebar.radio("Select a page", list(pages.keys()))
    pages[selected_page]()  # Call the corresponding function based on the selection

# Run the app
if __name__ == "__main__":
    main()
