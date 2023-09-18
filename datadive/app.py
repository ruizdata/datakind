import pandas as pd
import numpy as np
import openai
import streamlit as st
from app_funcs import generate_category_embeddings, load_data, clean_embeddings

st.title('An app that maps consumption categories from any dataset to IMF categories ')

st.markdown(""" 
            ## How do I use this app? 
            Say you have a dataset of household consumption for a country. You want to compare household consumption in this country to another country. If both countries used the IMF consumption categories, no problem. 
            
            But this is often not the case! What do you do if your dataset uses different categories to those used by the IMF?
            
            You could manually map your categories to the IMF ones, but what if we could do this automatically using AI?
            
            This app does exactly that!
            
            Simply upload your dataset and select the column that has category data in it. We'll use AI to identify the IMF categories most similar to the ones in your data. You can then download your data back again with the IMF categories added.            
            """
            )

st. markdown(
    """
    #### A note about OpenAI API keys
    We're using an AI model by OpenAI. It costs money to run, but in our case that amount is totally trivial (less than 1 US cent). Nonetheless, you'll need to get an OpenAI account, generate an API key and attach a credit card to it. This app does not store any data.
    Get an OpenAI account here: t.ly/Jrt8D then follow instructions to generate an API key.  
    """
)

user_input = st.text_input("You'll need to put your OpenAI API key here:")

if user_input:
    openai.api_key = user_input
else:
    openai.api_key = "REDACTED" 

uploaded_file = st.file_uploader("Upload your data here as a CSV", type="csv")

if uploaded_file:
    
    df = load_data(uploaded_file)
    
    st.markdown("Here is what we think your data looks like:")
    st.write(df.head())
    
    selected_column = st.selectbox('Select the column that represents the consumption categories:', df.columns)
    
    # Assign the selected column to a variable
    categories = list(df[selected_column].unique())
    
    show_categories = st.checkbox("Show unique consumption categories?")

    if show_categories:
        st.markdown("These are the unique categories you have")
        st.write(categories)
    
    # Now we provide functionality to get embeddings. The issue here is that we need to be careful about how often we call this function because it technically costs money to ping the OpenAI embeddings API
    
    embeddings_generated = False # default behaviour
    
    if st.button("Generate AI embeddings for categories (costs money)"):
        category_embeddings = generate_category_embeddings(categories)
        category_embeddings_clean = clean_embeddings(category_embeddings)
        
        embeddings_generated = True
    
    if st.button("Wipe embeddings (do this before regenerating embeddings)"):
        embeddings_generated = False
    
    if embeddings_generated:
        st.write(category_embeddings_clean)
    
    
    
# todo list
# - Reads in category and sub category embedding csvs
# - func to calculate similarity between data data embeddings and IMF embeddings
# - shows a table of data categories, and the top 3 IMF categorys by similarity as well as their scores.
# - option to download 

