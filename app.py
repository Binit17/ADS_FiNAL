import streamlit as st
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the trained BERTopic model
@st.cache_resource
def load_model():
    # Replace with the path where you saved the BERTopic model
    return BERTopic.load("Trained_models/bertopic_news_model")

# Load the model
topic_model = load_model()

# Function to display the topic hierarchy
def display_topic_hierarchy():
    st.subheader("Topic Hierarchy (Dendrogram)")
    fig = topic_model.visualize_hierarchy()
    st.plotly_chart(fig)

# Function to display the topic distribution
def display_topic_distribution():
    st.subheader("Topic Distribution")
    topics, _ = topic_model.get_topic_info(), topic_model.get_topics()
    topic_counts = topic_model.get_topic_info()
    
    # Plotting topic distribution as a boxplot
    fig, ax = plt.subplots()
    sns.boxplot(data=topics, x='Topic', y='Count', ax=ax)
    st.pyplot(fig)

# Function to display the topic frequencies
def display_topic_frequencies():
    st.subheader("Topic Frequencies")
    topics, _ = topic_model.get_topic_info(), topic_model.get_topics()
    fig = topic_model.visualize_topics()
    st.plotly_chart(fig)

# Function to display a topic's words
def display_topic_words(topic_id):
    st.subheader(f"Top words for Topic {topic_id}")
    words = topic_model.get_topic(topic_id)
    st.write(words)

# Interactive topic selection for words visualization
def display_topic_words_interactive():
    topics = topic_model.get_topic_info()
    topic_id = st.selectbox("Select Topic", topics['Topic'].unique())
    if topic_id is not None:
        display_topic_words(topic_id)

# Main app
def main():
    st.title("BERTopic Visualization Dashboard")

    # Display available visualizations
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ("Topic Hierarchy", "Topic Distribution", "Topic Frequencies", "Topic Words"))

    if page == "Topic Hierarchy":
        display_topic_hierarchy()

    elif page == "Topic Distribution":
        display_topic_distribution()

    elif page == "Topic Frequencies":
        display_topic_frequencies()

    elif page == "Topic Words":
        display_topic_words_interactive()

# Run the app
if __name__ == "__main__":
    main()
