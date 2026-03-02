import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(filepath):
    """Load the projects dataset."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

def prepare_text(row):
    """Combine features into a single text string for TF-IDF."""
    return f"{row['domain']} {row['level']} {row['tools_required']} {row['time_required']}"

def build_recommender(df):
    """Build the TF-IDF matrix and vectorizer."""
    if df.empty:
        return None, None
    df['combined_features'] = df.apply(prepare_text, axis=1)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    return vectorizer, tfidf_matrix

def skill_gap_analyzer(user_tools, required_tools_str):
    """Find missing tools for a recommended project."""
    if pd.isna(required_tools_str):
        return []
    # Split required tools by comma
    required_tools = [t.strip() for t in required_tools_str.split(',')]
    user_tools_lower = [t.lower() for t in user_tools]
    missing_tools = [t for t in required_tools if t.lower() not in user_tools_lower]
    return missing_tools

def get_recommendations(user_profile, df, vectorizer, tfidf_matrix, top_n=5):
    """
    Get top N project recommendations based on user profile.
    """
    if df.empty or vectorizer is None:
        return []

    # Map profile to text
    tools_str = " ".join(user_profile['tools_known'])
    user_text = f"{user_profile['domain']} {user_profile['level']} {tools_str} {user_profile['time_available']}"
    
    user_vector = vectorizer.transform([user_text])
    
    # Compute similarities
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Get top matching indices
    similar_indices = cosine_sim.argsort()[::-1][:top_n]
    
    recommendations = []
    for idx in similar_indices:
        score = cosine_sim[idx]
        if score > 0.0:  # Only recommend if there is some match
            row = df.iloc[idx].to_dict()
            missing_tools = skill_gap_analyzer(user_profile['tools_known'], row['tools_required'])
            
            recommendations.append({
                'project': row,
                'match_score': round(score * 100, 2),
                'missing_tools': missing_tools
            })
            
    return recommendations
