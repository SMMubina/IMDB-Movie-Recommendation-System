import streamlit as st  
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit page config
st.set_page_config(page_title="IMDb Movie Recommender", layout="wide")

# Load data and models
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('cleaned_movies.csv')

    # Fix NaNs and ensure string type for TF-IDF input
    data['Cleaned_Storyline'] = data['Cleaned_Storyline'].fillna('').astype(str)
    
    if not (os.path.exists('tfidf_model.pkl') and os.path.exists('cosine_sim.pkl')):
        st.warning("Generating recommendation models...")

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(data['Cleaned_Storyline'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        with open('tfidf_model.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        with open('cosine_sim.pkl', 'wb') as f:
            pickle.dump(cosine_sim, f)
    else:
        with open('tfidf_model.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('cosine_sim.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)

    return data, tfidf, cosine_sim

data, tfidf, cosine_sim = load_and_preprocess_data()

# Recommendation logic
def recommend_movies(movie_name, data, cosine_sim, top_n=5):
    try:
        matches = data[data['Movie Name'].str.lower() == movie_name.lower()]
        if len(matches) == 0:
            return None
        movie_index = matches.index[0]
        sim_scores = list(enumerate(cosine_sim[movie_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return data.iloc[[i[0] for i in sim_scores[1:top_n+1]]]
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

# UI layout
st.title("🎬 IMDb Movie Recommendation System")
st.write("Find movies with similar storylines, styled like IMDb cards.")

search_option = st.radio("Search by:", ["Movie Title", "Custom Storyline"], horizontal=True)

if search_option == "Movie Title":
    movie_query = st.selectbox(
        "Select a movie:", 
        data['Movie Name'].sort_values().unique(),
        index=None,
        placeholder="Type or select a movie..."
    )

    if st.button("🔍 Search"):
        if movie_query:
            st.subheader(f"Movies similar to: {movie_query}")
            recommendations = recommend_movies(movie_query, data, cosine_sim)

            if recommendations is not None:
                for _, row in recommendations.iterrows():
                    with st.container():
                        cols = st.columns([1, 4])
                        with cols[0]:
                            poster_url = row.get('Image URL')
                            if pd.notna(poster_url) and isinstance(poster_url, str) and poster_url.startswith("http"):
                                st.image(poster_url, width=150)  # Fixed width for consistency
                            else:
                                st.image('https://via.placeholder.com/150x225?text=No+Image', width=150)
                        with cols[1]:
                            rating = row.get('Rating', 'N/A')
                            year = row.get('Year', 'N/A')
                            duration = row.get('Duration', 'N/A')
                            
                            st.markdown(f"""
                                <div style="padding-left: 10px;">
                                    <h3 style="margin-bottom: 5px; color: #5799EF;">{row['Movie Name']}</h3>
                                    <div style="display: flex; gap: 15px; align-items: center; margin-bottom: 8px;">
                                        <span style="font-weight: bold;">⭐ {rating}</span>
                                        <span style="font-weight: bold;">📅 {year}</span>
                                        <span style="font-weight: bold;">🕒 {duration}</span>
                                    </div>
                                    <p style="color: #000000; font-size: 14px; line-height: 1.4; margin-top: 8px;">{row['Storyline']}</p>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("Movie not found. Try another title.")

else:
    custom_story = st.text_area("Enter a movie storyline:")
    if st.button("🔍 Search"):
        if custom_story:
            st.subheader("Recommended movies based on your storyline:")
            custom_vec = tfidf.transform([custom_story])

            # Safe transformation of storylines - no NaN, all strings
            cleaned_storylines = data['Cleaned_Storyline'].fillna('').astype(str)
            sim_scores = cosine_similarity(custom_vec, tfidf.transform(cleaned_storylines))

            top_indices = sim_scores.argsort()[0][-5:][::-1]

            for idx in top_indices:
                row = data.iloc[idx]
                with st.container():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        poster_url = row.get('Image URL')
                        if pd.notna(poster_url) and isinstance(poster_url, str) and poster_url.startswith("http"):
                            st.image(poster_url, width=150)
                        else:
                            st.image('https://via.placeholder.com/150x225?text=No+Image', width=150)
                    with cols[1]:
                        rating = row.get('Rating', 'N/A')
                        year = row.get('Year', 'N/A')
                        duration = row.get('Duration', 'N/A')
                        
                        st.markdown(f"""
                            <div style="padding-left: 10px;">
                                <h3 style="margin-bottom: 5px; color: #5799EF;">{row['Movie Name']}</h3>
                                <div style="display: flex; gap: 15px; align-items: center; margin-bottom: 8px;">
                                    <span style="font-weight: bold;">⭐ {rating}</span>
                                    <span style="font-weight: bold;">📅 {year}</span>
                                    <span style="font-weight: bold;">🕒 {duration}</span>
                                </div>
                                <p style="color: #000000; font-size: 14px; line-height: 1.4; margin-top: 8px;">{row['Storyline']}</p>
                            </div>
                        """, unsafe_allow_html=True)

# Custom CSS Styling
st.markdown("""
<style>
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
        border-radius: 8px;
        margin-bottom: 20px;
    }
    h3 {
        font-size: 22px;
        margin-top: 0;
    }
    .stMarkdown p {
        margin-bottom: 0.5rem;
    }
    .stContainer {
        background-color: #0c0c0c;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .st-emotion-cache-1jicfl2 {
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
