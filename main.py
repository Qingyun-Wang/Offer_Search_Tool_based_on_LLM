from engine import * 
from database import *
import numpy as np

if __name__ == "__main__":
    offer_data = query_db()
    input_text = input("what do you want to search: ")
    result = run_engine(offer_data, input_text, 20)
    print(result)
