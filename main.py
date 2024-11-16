from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel #defines what all features/columns a model is expecting.
import pickle
import numpy as np #for preprocessing
import pandas as pd #for preprocessing
from io import StringIO
import requests

# Initialize FastAPI application
app = FastAPI() #constructor

# Define a Pydantic model for the input data schema expected by the churn prediction model
class IrisClass(BaseModel):
   SepalLengthCm:float
   SepalWidthCm:float
   PetalLengthCm:float
   PetalWidthCm:float

# Basic root endpoint to check if the API is running
#decorator- modifies the behaviour of the function.
@app.get("/") 
async def root(): #async-doesn't block other requests coming in.
    return {"message": "Hello World"}

# Endpoint to predict churn based on a single user's data
@app.post("/predict")
async def predict_iris(pk: IrisClass):
    """
    Receives input data for a single user, transforms it to the required format,
    sends it to the machine learning model API, and returns the model's response.
    """
    # Convert Pydantic model to dictionary
    data = pk.dict()

    # Format input data as a 2D array (list of lists) for the model
    data_in = [[
        data['SepalLengthCm'],
        data['SepalWidthCm'],
        data['PetalLengthCm'],
        data['PetalWidthCm']
    ]]

    # Print data_in for debugging purposes
    print("Input data for model:", data_in)

    # Define the endpoint for the machine learning model
    endpoint = "http://localhost:8888/invocations"
    
    # Create a request payload for the model
    inference_request = {"dataframe_records": data_in}
    
    # Send the request to the ML model and receive response
    response = requests.post(endpoint, json=inference_request)
    
    # Return the response text from the model as the API response
    return response.text

# Endpoint to perform batch predictions from a CSV file
@app.post("/files/")
async def batch_prediction(file: bytes = File(...)):
    """
    Accepts a CSV file, processes its contents to match the input format of the ML model,
    and returns predictions for each row in the CSV file.
    """
    # Convert uploaded file bytes to a string
    s = str(file, 'utf-8')
    
    # Use StringIO to read the CSV data as a pandas DataFrame
    data = StringIO(s)
    df = pd.read_csv(data)
    
    # Convert the DataFrame to a list of lists (expected format for model input)
    lst = df.values.tolist()
    
    # Create a request payload for the batch data
    inference_request = {"dataframe_records": lst}
    
    # Define the model's endpoint
    endpoint = "http://localhost:8888/invocations"
    
    # Send the request to the ML model and receive the batch response
    response = requests.post(endpoint, json=inference_request)
    
    # Print the response text for debugging
    print("Batch response:", response.text)
    
    # Return the response text from the model as the API response
    return response.text #TO SEND THE RESPONSE

# To start the FastAPI server, run the following command in the terminal:
# uvicorn main:app --reload
