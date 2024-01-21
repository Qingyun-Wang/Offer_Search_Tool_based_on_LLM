Instuction of using this tool: (You can also click here for [web application](https://offer-search-tool-based-on-llm.onrender.com)):

1) create and install a virtual env using the requirements.txt file (linux or mac):
    python3 -m venv path/to/new/env
    source path/to/new/env/bin/activate
    pip install -r requirements.txt
2) run the tool:
    streamlit run main.py
3) follow the prompt and enter your search

we preprocessed the raw data and generated the "processed_data.pkl" file, 
the engine will compare the inputs with data in processed_data and return offers, brand, retailers and combined score

Please check the Fetch_takehome_report
