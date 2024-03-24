
# Q&A bot: Question and Answer System Based on Google Palm LLM and Langchain for an E-commerce company

This is an end to end LLM project based on Google Palm and Langchain. We are building a Q&A system for an e-commerce company. This system will provide a streamlit based user interface for users where they can ask questions and get answers. 

## Project Highlights

- We will build an LLM based question and answer system that can reduce dependency on human staff.
- Users should be able to use this application to ask questions directly and get answers within seconds

## You will learn following,
  - Langchain + Google Palm: LLM based Q&A
  - Streamlit: UI
  - Huggingface instructor embeddings: Text embeddings
  - FAISS: Vector databse

## Installation

1.Clone this repository to your local machine using:

```bash
  git clone <git repo link>
```
2.Navigate to the project directory:

```bash
  cd <file path>
```

3.Prepare a csv file named sample_faqs.csv with two columns called prompt and response. Collect few Q&A, paste your questions in column, prompt and answers in response.

4.Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
5.Acquire an api key through makersuite.google.com and put it in .env file

```bash
  GOOGLE_API_KEY="your_api_key_here"
```
## Usage


1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- To create a knowledebase of FAQs, click on Create Knolwedge Base button. It will take some time before knowledgebase is created so please wait.

- Once knowledge base is created you will see a directory called faiss_index in your current folder

- Now you are ready to ask questions. Type your question in Question box and hit Enter

## Sample Questions
  - Do you offer EMI payments?
  - How do I cancel the order, I have placed??
  - How do I create a Return Request??

## Project Structure

- main.py: The main Streamlit application script.
- langchain_helper.py: This has all the langchain code
- requirements.txt: A list of required Python packages for the project.
- .env: Configuration file for storing your Google API key.
