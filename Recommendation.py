import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
# Check if dataset exists
if not os.path.exists("Food_Recipe.csv"):
    st.error("Food_Recipe.csv dataset not found. Please upload the dataset to continue.")
    st.info("The application requires a CSV file with columns: name, ingredients_name, cuisine, diet, course, instructions")
    st.stop()
try:
    # Load and clean data
    df = pd.read_csv("Food_Recipe.csv")
    
    # Check if required columns exist
    required_columns = ['name', 'ingredients_name', 'cuisine', 'diet', 'course', 'instructions']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns in dataset: {', '.join(missing_columns)}")
        st.info("Required columns: name, ingredients_name, cuisine, diet, course, instructions")
        st.stop()
    
    # Select and clean data
    df = df[required_columns]
    df.dropna(subset=['ingredients_name'], inplace=True)
    df['ingredients_name'] = df['ingredients_name'].astype(str).str.strip().str.lower()
    df = df[df['ingredients_name'].str.len() > 5].reset_index(drop=True)
    
    if len(df) == 0:
        st.error("No valid recipes found in the dataset after cleaning.")
        st.info("Please ensure the dataset contains recipes with valid ingredient information.")
        st.stop()
    
    # TF-IDF and Cosine Similarity
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['ingredients_name'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
except Exception as e:
    st.error(f"Error loading or processing dataset: {str(e)}")
    st.info("Please check that your Food_Recipe.csv file is properly formatted and contains the required data.")
    st.stop()
# Main application title
st.title("Food Recipe Recommendation System")
# Sidebar Filters
st.sidebar.header("Filter Options")
# Get unique values for filters, handling potential NaN values
cuisine_options = ['All Cuisines'] + sorted([c for c in df['cuisine'].dropna().unique() if pd.notna(c)])
diet_options = ['All Diets'] + sorted([d for d in df['diet'].dropna().unique() if pd.notna(d)])
course_options = ['All Courses'] + sorted([c for c in df['course'].dropna().unique() if pd.notna(c)])
recipe_options = [''] + sorted([r for r in df['name'].unique() if pd.notna(r)])
cuisine = st.sidebar.selectbox("Select Cuisine", cuisine_options)
diet = st.sidebar.selectbox("Select Diet", diet_options)
course = st.sidebar.selectbox("Select Course", course_options)
recipe = st.sidebar.selectbox("Select a Recipe (Optional)", recipe_options)
ingredient = st.sidebar.text_input("Key Ingredient (Optional)", placeholder="e.g., rice, tomato, coconut")
num_results = st.sidebar.slider("Number of Results", 1, 10, 5)
# Add some information about the system
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("-> Filter recipes by cuisine, diet, or course")
st.sidebar.markdown("-> Search by key ingredients")
st.sidebar.markdown("-> Get similar recipes based on ingredients")
st.sidebar.markdown("-> Uses TF-IDF and cosine similarity")
# Main recommendation logic
if st.button(" Get Recommendations", type="primary"):
    # Apply filters
    filtered_df = df.copy()
    
    if cuisine != 'All Cuisines':
        filtered_df = filtered_df[filtered_df['cuisine'] == cuisine]
    
    if diet != 'All Diets':
        filtered_df = filtered_df[filtered_df['diet'] == diet]
    
    if course != 'All Courses':
        filtered_df = filtered_df[filtered_df['course'] == course]
    
    if ingredient:
        filtered_df = filtered_df[filtered_df['ingredients_name'].str.contains(ingredient.lower(), na=False)]
    st.subheader("Food Recommendation Results")
    
    # Handle empty results
    if len(filtered_df) == 0:
        st.warning("No recipes found matching your criteria. Try adjusting your filters.")
        st.info("**Suggestions:**\n- Try selecting 'All' for some filter options\n- Check your ingredient spelling\n- Use more general ingredient terms")
    






















    
    # Recipe-based similarity recommendations
    elif recipe:
        try:
            recipe_idx = df[df['name'] == recipe].index[0]
            sim_scores = list(enumerate(cosine_sim[recipe_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_results + 1]
            st.success(f"Found {len(sim_scores)} similar recipes to '{recipe}'")
            
            for i, (idx, score) in enumerate(sim_scores, 1):
                r = df.iloc[idx]
                
                # Create a container for each recipe
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {r['name']}**")
                        st.markdown(f"**Cuisine:** {r['cuisine']} | **Course:** {r['course']}")
                        
                        # Show complete ingredients
                        st.markdown(f"**Ingredients:** {r['ingredients_name']}")
                    
                    with col2:
                        st.metric("Similarity", f"{score:.3f}")
                    
                    st.markdown("---")
                        
        except (IndexError, KeyError):
            st.error(f"Recipe '{recipe}' not found in the database.")
            st.info("Please select a recipe from the dropdown menu or leave it empty for general recommendations.")
    
    # General filtered recommendations
    else:
        results = filtered_df.head(num_results)
        st.success(f"Showing {len(results)} recipes matching your criteria")
        
        for i, (_, r) in enumerate(results.iterrows(), 1):
            with st.container():
                st.markdown(f"**{i}. {r['name']}**")
                
                # Create columns for better layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Cuisine:** {r['cuisine']} | **Diet:** {r['diet']} | **Course:** {r['course']}")
                
                # Show complete ingredients
                st.markdown(f"**Ingredients:** {r['ingredients_name']}")
                
                st.markdown("---")
    # Statistics section
    st.markdown("### Database Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Recipes", len(df))
    
    with col2:
        st.metric("Matching Filters", len(filtered_df))
    
    with col3:
        results_shown = min(num_results, len(filtered_df)) if len(filtered_df) > 0 else 0
        st.metric("Results Displayed", results_shown)
# Display initial information when no search is performed
else:
    st.markdown("### Welcome to the Recipe Recommendation System!")
    st.markdown("""
    **Dataset Overview:**
    """)
    
    # Show dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recipes", len(df))
    
    with col2:
        unique_cuisines = len(df['cuisine'].dropna().unique())
        st.metric("Unique Cuisines", unique_cuisines)
    
    with col3:
        unique_diets = len(df['diet'].dropna().unique())
        st.metric("Diet Types", unique_diets)
    
    with col4:
        unique_courses = len(df['course'].dropna().unique())
        st.metric("Course Types", unique_courses)
    
    # Show sample recipes
    st.markdown("### Sample Recipes in Database")
    sample_recipes = df.sample(min(3, len(df)))
    
    for _, recipe in sample_recipes.iterrows():
        with st.expander(f"{recipe['name']}"):
            st.markdown(f"**Cuisine:** {recipe['cuisine']}")
            st.markdown(f"**Diet:** {recipe['diet']}")
            st.markdown(f"**Course:** {recipe['course']}")
            st.markdown(f"**Ingredients:** {recipe['ingredients_name']}")