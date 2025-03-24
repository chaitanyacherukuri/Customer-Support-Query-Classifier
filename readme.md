# Customer Support Query Classifier

This project is an AI-powered customer support query classifier built using Streamlit, LangChain, and Groq. It classifies customer queries into **Billing, Tech Support, or Sales** departments and generates appropriate responses.

## Setup

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Groq](https://groq.com/)

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/chaitanyacherukuri/customer-support-query-classifier.git
    cd customer-support-query-classifier
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Set up the Groq API key:

    - Open `.streamlit/secrets.toml` and add your Groq API key:

    ```toml
    GROQ_API_KEY = "your_groq_api_key"
    ```

## Usage

1. Run the Streamlit app:

    ```sh
    streamlit run Customer_Query_Classifier.py
    ```

2. Open your web browser and go to `http://localhost:8501` to access the app.

## Features

- **Query Classification**: Classifies customer queries into Billing, Tech Support, or Sales departments.
- **Response Generation**: Generates appropriate responses based on the classified department.
- **Sample Questions**: Provides sample questions for testing the classifier.
- **Confidence Score**: Displays the confidence score of the classification.

## Dependencies

- langchain
- langchain-core
- langchain-community
- langchain-groq
- langgraph
- streamlit