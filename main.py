import os
import pandas as pd
from engine import * 

if __name__ == "__main__":
    PATH = os.getcwd()
    training = pd.read_csv(PATH+'/data/processed_data.csv')
    input_text = input("what do you want to search: ")
    run_engine(training, input_text, 10)
