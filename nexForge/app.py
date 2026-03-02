import streamlit as st
import pandas as pd
import os
from src.recommender import load_data, build_recommender, get_recommendations

# Page Configuration
st.set_page_config(
    page_title="nexForgeAI - ML Project Recommender",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styled cards
st.markdown("""
<style>
.project-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #0066cc;
}
/* Basic dark mode support adaptation */
[data-testid="stAppViewContainer"] .project-card {
    background-color: rgba(128, 128, 128, 0.1);
}
.tool-tag {
    display: inline-block;
    background-color: rgba(100, 100, 100, 0.2);
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.8rem;
    margin-right: 5px;
    margin-bottom: 5px;
}
.missing-tool-tag {
    display: inline-block;
    background-color: rgba(255, 0, 0, 0.1);
    color: #ff4b4b;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.8rem;
    margin-right: 5px;
    margin-bottom: 5px;
    border: 1px solid rgba(255, 0, 0, 0.3);
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🚀 nexForgeAI")
    st.markdown("### Personalized AI/ML Project Recommender")
    st.write("Discover your next big project based on your skills, interests, and availability.")

    # Load data and build model
    data_path = "projects.csv"
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please ensure projects.csv exists.")
        return

    df = load_data(data_path)
    if df.empty:
        st.error("Failed to load dataset or dataset is empty.")
        return

    vectorizer, tfidf_matrix = build_recommender(df)

    # Sidebar Inputs
    st.sidebar.header("👤 Your Profile")
    
    domain = st.sidebar.selectbox(
        "Preferred Domain",
        (
            "Healthcare", "FinTech", "AI", "Education", "Logistics",
            "Hardware", "IoT", "Hybrid", "Web", "Mobile", "Game Dev"
        )
    )
    
    level = st.sidebar.selectbox(
        "Skill Level",
        ("Beginner", "Intermediate", "Advanced")
    )
    
    tools_known = st.sidebar.multiselect(
        "Tools & Technologies Known",
        options=[
            # AI / ML
            "Python", "ML", "Deep Learning", "NLP",
            # Cloud & DevOps
            "AWS", "Docker",
            # Mobile & UI
            "Flutter",
            # Data
            "SQL",
            # Hardware & IoT
            "Arduino", "Raspberry Pi", "Electronics", "MQTT", "FPGA", "C++", "VHDL",
            # Web
            "JavaScript", "React", "HTML", "CSS",
            # Game Dev
            "Unity", "C#", "Pygame",
        ],
        default=["Python"]
    )
    
    time_available = st.sidebar.selectbox(
        "Time Available",
        ("3 days", "1 week", "1 month")
    )

    get_recs_btn = st.sidebar.button("Get Recommendations 🎯", type="primary", use_container_width=True)

    if get_recs_btn:
        if not tools_known:
            st.warning("Please select at least one tool you know.")
            return
            
        with st.spinner("Analyzing your profile and finding the best projects..."):
            user_profile = {
                'domain': domain,
                'level': level,
                'tools_known': tools_known,
                'time_available': time_available
            }
            
            recommendations = get_recommendations(user_profile, df, vectorizer, tfidf_matrix, top_n=5)
            
            if not recommendations:
                st.info("No projects found matching your criteria. Try adjusting your preferences.")
            else:
                st.subheader("Top Recommended Projects For You")
                
                for i, rec in enumerate(recommendations):
                    project = rec['project']
                    match_score = rec['match_score']
                    missing_tools = rec['missing_tools']
                    difficulty = int(project['difficulty_score'])
                    
                    # Create styled card using HTML/CSS and Streamlit components
                    st.markdown(f"""
                    <div class="project-card">
                        <h3>{i+1}. {project['title']}</h3>
                        <p>{project['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="display:flex; gap:30px; flex-wrap:wrap; margin-bottom:10px;">
                        <div><span style="font-size:0.78rem; color:gray;">Match Score</span><br><strong style="font-size:1.2rem;">{match_score}%</strong></div>
                        <div><span style="font-size:0.78rem; color:gray;">Domain</span><br><strong style="font-size:1.2rem;">{project['domain']}</strong></div>
                        <div><span style="font-size:0.78rem; color:gray;">Skill Level</span><br><strong style="font-size:1.2rem;">{project['level']}</strong></div>
                        <div><span style="font-size:0.78rem; color:gray;">Time Required</span><br><strong style="font-size:1.2rem;">{project['time_required']}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                        
                    # Difficulty Progress Bar
                    st.caption(f"Difficulty Score: {difficulty}/100")
                    st.progress(difficulty / 100.0)
                    
                    # Tools Required
                    req_tools = [t.strip() for t in str(project['tools_required']).split(',')]
                    tools_html = "<strong>Tools Required:</strong><br>"
                    for tool in req_tools:
                        tools_html += f'<span class="tool-tag">{tool}</span>'
                    st.markdown(tools_html, unsafe_allow_html=True)
                    
                    # Skill Gap Analyzer
                    if missing_tools:
                        gap_html = "<strong>⚠️ Skill Gap to Bridge:</strong><br>"
                        for tool in missing_tools:
                            gap_html += f'<span class="missing-tool-tag">{tool}</span>'
                        st.markdown(gap_html, unsafe_allow_html=True)
                    else:
                        st.success("✨ You have all the required tools for this project!")
                        
                    # Dataset and Architecture
                    st.markdown(f"**Architecture:** {project['architecture']} | [Dataset Link]({project['dataset_link']})")
                    st.divider()

if __name__ == "__main__":
    main()
