from engine import * 
from database import *
import numpy as np
import streamlit as st


def main():
    offer_data = query_db()

    st.title("Search Offers by input product, brand or retailer")
    
    # Create a text input widget in Streamlit
    input_text = st.text_input("What do you want to search:")
    
    # Run the search engine when input is provided
    if input_text:
        result = run_engine(offer_data, input_text, 10)
        st.write(result)  # Display the result

if __name__ == "__main__":
    main()