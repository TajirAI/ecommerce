import os
import json
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

GPT_API_KEY = st.secrets["api"]["key"]  # or st.secrets.api.key
if not GPT_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

############################################
# Step 1: Query Classification
############################################
def classify_question(question: str) -> str:
    prompt_template = """
You are an assistant that categorizes user queries regarding our ecommerce platform into one of the following categories:
1. General Query: Questions about products, features, pricing, etc.
2. Order Request: Requests to place an order.
3. Order Update: Inquiries regarding the status of an existing order.
Based on the user question provided below, reply with exactly one of the following words: "General Query", "Order Request", or "Order Update". Do not include any extra text.
if user asked the question in Bengali language then answer in Bengali.
User question: {question}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, openai_api_key=GPT_API_KEY)
    response = model.call_as_llm(prompt.format(question=question))
    return response.strip()

############################################
# Step 2: Build & Load the Products Vector Store from CSV
############################################
def load_products_vectorstore(csv_path: str = "products.csv", vectorstore_path: str = "products_faiss_index"):
    if os.path.exists(vectorstore_path):
        embeddings = OpenAIEmbeddings(openai_api_key=GPT_API_KEY)
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    else:
        df = pd.read_csv(csv_path)
        docs = []
        for _, row in df.iterrows():
            content = f"Product: {row['Name']}\nPrice: {row['Price']}\nCategory: {row['Category']}"
            docs.append(Document(page_content=content))
        embeddings = OpenAIEmbeddings(openai_api_key=GPT_API_KEY)
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        vectorstore.save_local(vectorstore_path)
        return vectorstore

############################################
# Step 3: Process General Queries Using RAG on Product Data with Conversation History
############################################
def process_general_query(question: str) -> str:
    vectorstore = load_products_vectorstore()
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(question)
    
    # Load CSV to get all unique product categories.
    df = pd.read_csv("products.csv")
    categories = df["Category"].unique().tolist()
    categories_str = ", ".join(categories)
    
    # Retrieve conversation history from memory.
    memory_vars = st.session_state["memory"].load_memory_variables({})
    history = memory_vars.get("chat_history", "")
    
    prompt_template = """
You are a helpful assistant that uses the following conversation context along with product details to answer the customer's query.
If a product is available with us, please fetch its specifications from the internet and give them to the client. But if a product is not in the context,
DO NOT SHARE ITS SPECIFICATIONS; just say that it's not available.
if user asked the question in Bengali language then answer in Bengali.
If the user generally asks for products or categories, you should provide all product categories.
Conversation History:
{history}

Product Details:
{context}

All Product Categories: {categories}

Customer Query: {question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question", "categories"])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=GPT_API_KEY)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({
        "input_documents": docs, 
        "question": question, 
        "categories": categories_str,
        "history": history
    }, return_only_outputs=True)
    return response["output_text"]

############################################
# Step 4: Process Order Request with Interactive Flow
############################################
def process_order_request(question: str) -> str:
    # Check for cancellation command.
    if question.strip().lower() in ["cancel", "cancel order", "stop"]:
         st.session_state["order_in_progress"] = False
         st.session_state["order_details"] = {}
         return "Order has been cancelled."
    
    # If order details are already confirmed, finalize and store the order.
    if st.session_state.get("order_details", {}).get("order_status") == "confirmed":
         confirmed_details = st.session_state["order_details"]
         order_info = {
             "Name": confirmed_details.get("Name"),
             "Address": confirmed_details.get("Address"),
             "Product": confirmed_details.get("Product"),
             "Quantity": confirmed_details.get("Quantity"),
             "Status": "received"
         }
         order_df = pd.DataFrame([order_info])
         if os.path.exists("orders.csv"):
             order_df.to_csv("orders.csv", mode='a', header=False, index=False)
         else:
             order_df.to_csv("orders.csv", index=False)
         st.session_state["order_in_progress"] = False
         st.session_state["order_details"] = {}
         return f"Order confirmed! Details: {order_info}"
    
    # Check if the current user input is a confirmation phrase.
    if question.strip().lower() in ["go ahead", "confirm", "okay thanks"]:
         required_keys = ["Name", "Address", "Product", "Quantity", "order_status"]
         if all(k in st.session_state.get("order_details", {}) for k in required_keys):
              confirmed_details = st.session_state["order_details"]
              if confirmed_details.get("order_status") == "confirmed":
                  order_info = {
                      "Name": confirmed_details.get("Name"),
                      "Address": confirmed_details.get("Address"),
                      "Product": confirmed_details.get("Product"),
                      "Quantity": confirmed_details.get("Quantity"),
                      "Status": "received"
                  }
                  order_df = pd.DataFrame([order_info])
                  if os.path.exists("orders.csv"):
                      order_df.to_csv("orders.csv", mode='a', header=False, index=False)
                  else:
                      order_df.to_csv("orders.csv", index=False)
                  st.session_state["order_in_progress"] = False
                  st.session_state["order_details"] = {}
                  return f"Order confirmed! Details: {order_info}"
         # If details are incomplete, we fall through to ask the LLM for more information.
    
    # Retrieve conversation history.
    memory_vars = st.session_state["memory"].load_memory_variables({})
    history = memory_vars.get("chat_history", "")
    
    prompt_template = """
You are an assistant that helps complete a customer's order interactively. Based on the conversation history below and the customer's latest input, extract the following order details if provided:
- Customer's Name
- Customer's Address
- Product Name (if already discussed in previous conversation, use that information)
- Quantity
if user asked the question in Bengali language then answer in Bengali.
If some details are missing, ask a follow-up question in a natural conversational manner to get that information.
If all details are provided, output a confirmation in JSON format with the following keys:
"Name": customer's name,
"Address": customer's address,
"Product": product name,
"Quantity": quantity,
"order_status": "confirmed"

If the customer types "cancel" at any point, cancel the order process.

Conversation History:
{history}

Current Customer Input:
{question}

Existing Order Details (if any):
{order_details}

Respond:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "question", "order_details"])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=GPT_API_KEY)
    llm_response = model.call_as_llm(prompt.format(
        history=history,
        question=question,
        order_details=json.dumps(st.session_state.get("order_details", {}))
    ))
    
    try:
         response_json = json.loads(llm_response)
         if response_json.get("order_status") == "confirmed":
              order_info = {
                  "Name": response_json.get("Name", ""),
                  "Address": response_json.get("Address", ""),
                  "Product": response_json.get("Product", ""),
                  "Quantity": response_json.get("Quantity", ""),
                  "Status": "received"
              }
              order_df = pd.DataFrame([order_info])
              if os.path.exists("orders.csv"):
                  order_df.to_csv("orders.csv", mode='a', header=False, index=False)
              else:
                  order_df.to_csv("orders.csv", index=False)
              st.session_state["order_in_progress"] = False
              st.session_state["order_details"] = {}
              return f"Order confirmed! Details: {order_info}"
         else:
              st.session_state["order_details"].update(response_json)
              st.session_state["order_in_progress"] = True
              return llm_response
    except json.JSONDecodeError:
         st.session_state["order_in_progress"] = True
         return llm_response


############################################
# Step 5: Process Order Update with Interactive Flow
############################################
def process_order_update(question: str) -> str:
    # Check for cancellation command.
    if question.strip().lower() in ["cancel", "cancel update", "stop"]:
         st.session_state["order_update_in_progress"] = False
         st.session_state["order_update_details"] = {}
         return "Order update process has been cancelled."

    # Retrieve conversation history.
    memory_vars = st.session_state["memory"].load_memory_variables({})
    history = memory_vars.get("chat_history", "")
    
    # Retrieve any existing update details from session.
    update_details = st.session_state.get("order_update_details", {})

    prompt_template = """
You are an assistant that helps customers check the status of their orders. Based on the conversation history below and the customer's latest input, extract the following detail if provided:
- Customer's Name

If the detail is missing, ask a follow-up question in a natural conversational manner to obtain the missing information.
If the required detail is provided, output a confirmation in JSON format with the following keys:
"Name": customer's name,
"lookup_status": "ready"

if user asked the question in Bengali language then answer in Bengali.
If the customer types "cancel" at any point, cancel the update process.

Conversation History:
{history}

Current Customer Input:
{question}

Existing Order Update Details (if any):
{update_details}

Respond:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "question", "update_details"])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=GPT_API_KEY)
    llm_response = model.call_as_llm(prompt.format(history=history, question=question, update_details=json.dumps(update_details)))
    
    try:
        response_json = json.loads(llm_response)
        if response_json.get("lookup_status") == "ready" and response_json.get("Name"):
            customer_name = response_json.get("Name").strip()
            # Read orders.csv to look up the order(s)
            if not os.path.exists("orders.csv"):
                st.session_state["order_update_in_progress"] = False
                st.session_state["order_update_details"] = {}
                return "No orders found in our records."
            orders_df = pd.read_csv("orders.csv")
            # Filter by name (case-insensitive)
            matched_orders = orders_df[orders_df["Name"].str.lower() == customer_name.lower()]
            if matched_orders.empty:
                st.session_state["order_update_in_progress"] = False
                st.session_state["order_update_details"] = {}
                return f"No orders found for the name {customer_name}."
            else:
                # Instead of returning the full order details, return only the status(es)
                statuses = matched_orders["Status"].unique().tolist()
                if len(matched_orders) == 1:
                    return f"Your order status is: {statuses[0]}"
                else:
                    status_message = ', '.join(statuses)
                    return f"Multiple orders found for {customer_name}. The statuses are: {status_message}"
        else:
            st.session_state["order_update_details"].update(response_json)
            st.session_state["order_update_in_progress"] = True
            return llm_response
    except json.JSONDecodeError:
        st.session_state["order_update_in_progress"] = True
        return llm_response


############################################
# Step 6: Determine Which Process to Run Based on Query Type
############################################
def process_user_query(question: str) -> str:
    category = classify_question(question)
    if category == "General Query":
        return process_general_query(question)
    elif category == "Order Request":
        return process_order_request(question)
    elif category == "Order Update":
        return process_order_update(question)
    else:
        return f"Unable to classify the question. Received classification: {category}"

############################################
# Step 7: Main Streamlit Application with Chat History and Conversation Memory
############################################
def main():
    st.set_page_config(page_title="TajirAI EcomAssist", layout="wide")
    st.header("TajirAI EcomAssist")
    
    # Initialize conversation memory if not already present.
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=GPT_API_KEY),
            memory_key="chat_history"
        )
    
    # Initialize chat messages list.
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Initialize order request state.
    if "order_in_progress" not in st.session_state:
        st.session_state["order_in_progress"] = False
    if "order_details" not in st.session_state:
        st.session_state["order_details"] = {}

    # Initialize order update state.
    if "order_update_in_progress" not in st.session_state:
        st.session_state["order_update_in_progress"] = False
    if "order_update_details" not in st.session_state:
        st.session_state["order_update_details"] = {}
    
    # Display previous chat messages.
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input.
    user_query = st.chat_input(placeholder="Need quick help with products, orders, updates?")
    if user_query:
        # Append and display user message.
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process the user query.
        response = process_user_query(user_query)
        
        # Update conversation memory with the new interaction.
        st.session_state["memory"].save_context({"input": user_query}, {"output": response})
        
        # Append and display assistant response.
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()
