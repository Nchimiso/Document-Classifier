# Document-Classifier
Document classification system
### Set up

1. Install all necessary imports using the following line
    
    ```bash
    bash
    pip install PyMuPDF numpy gradio scikit-learn matplotlib
    ```
    
2. Load the necessary documents into the /data folder, ensuring that **within the data folder, new folders are created per document type**
    1. ie, the following is how the data file should look, and in this case, the only types of documents being classified are contract, invoice, and resume
        
        ![image.png](attachment:b1deda5f-aefe-4a5b-821e-4c78fc366de9:image.png)
        

---

### Usage

1. In the command line, run the following command to run the Python file and ensure that the necessary folder is set up, including the desired documents. 
    
    ```bash
    bash
    python yourfilename.py
    ```
    
2. You will now see a link in the command line. Click this link and use the gradio UI to submit documents to be classified.
