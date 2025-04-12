import os
import streamlit as st
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

#Intialize LLM
llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
#Intialize output parser
output_parser = StrOutputParser()

#Schema for structured output to use as routing logic
class CustomerQueryRouter(BaseModel):
    """Route the customer query to the appropriate department"""

    department: Literal["billing", "tech_support", "sales"] = Field(
        description="The department to route the customer query to"
    )

    confidence: float = Field(
        description="The confidence score between 0.0 and 1.0 for this classification."
    )

    reason: str = Field(
        description="Brief explanation of why this query belongs to the selected department."
    )

#Augment LLM with schema for structured output
llm_router = llm.with_structured_output(CustomerQueryRouter)

#State dictionary to track information throughout the workflow
class State(TypedDict):
    customer_query: str
    department: str
    confidence: float
    reason: str
    response: str

#Router node to classify the customer query
def classify_customer_query(state: State):
    """Determine which department should handle this customer query"""

    decision = llm_router.invoke(
        [
            SystemMessage(
                content="""You are an experienced customer service manager who has handled thousands of customer inquiries.
                
                Classify incoming customer messages into one of these departments:
                - billing: Payment issues, charges, invoices, subscription fees, refunds, payment methods, billing cycles
                - tech_support: Technical problems, error messages, software/hardware issues, setup help, troubleshooting steps
                - sales: New purchases, product comparisons, upgrade options, pricing questions, availability, features
                
                Your classification must be accurate as it determines which specialist will help the customer."""
            ),
            HumanMessage(
                content=state["customer_query"]
            )
        ]
    )

    return {"department": decision.department,
            "confidence": decision.confidence,
            "reason": decision.reason}

#Conditional edge to route the customer query to the appropriate department
def route_condition(state: State):
    """Determine the next node based on the department classification and confidence score"""

    if state["confidence"] < 0.7:
        return "unknown"
    
    return state["department"]

#Node to handle the billing department
def billing(state: State):
    """Generate a response for customer queries related to billing department"""

    response = llm.invoke(
        [
            SystemMessage(
                content="""You're Sarah from the Billing Department with 5 years of experience.

                Respond to the customer's billing question as if you're in a real-time chat. Be conversational yet professional.
                
                - Address their specific concern about payments, refunds, or subscriptions
                - Use a helpful, solution-oriented approach
                - If you need more information, ask a specific question
                - Mention the billing department's availability (Mon-Fri, 9am-6pm)
                - Include an offer to help with anything else
                
                Your goal is to make the customer feel their billing issue will be resolved quickly."""
            ),
            HumanMessage(
                content=state["customer_query"]
            )
        ]
    )

    return {"response": output_parser.invoke(response)}

#Node to handle the tech support department
def techsupport(state: State):
    """Generate a response for customer queries related to tech support department"""

    response = llm.invoke(
        [
            SystemMessage(
                content="""You're Alex from Technical Support with expertise in troubleshooting our products.
                
                Respond to the customer's technical issue as if you're chatting in real-time. Be conversational yet knowledgeable.
                
                - Show understanding of their frustration if they're experiencing problems
                - Provide clear, step-by-step troubleshooting advice when possible
                - Avoid overwhelming jargon while still being technically accurate
                - If you need more details about their setup or the issue, ask specific questions
                - Mention that complex issues can be escalated to our senior technical team if needed
                
                Your goal is to help them solve their technical problem efficiently while making them feel supported."""
            ),
            HumanMessage(
                content=state["customer_query"]
            )
        ]
    )

    return {"response": output_parser.invoke(response)}

#Node to handle the sales department
def sales(state: State):
    """Generate a response for customer queries related to sales department"""

    response = llm.invoke(
        [
            SystemMessage(
                content="""You're Jordan from the Sales Team with detailed knowledge of our products and services.
                
                Respond to the customer's sales inquiry as if you're in a real-time conversation. Be enthusiastic and helpful.
                
                - Show genuine interest in helping them find the right product/solution
                - Highlight relevant benefits rather than listing all features
                - If appropriate, mention current promotions or special offers
                - Ask clarifying questions about their needs if necessary
                - Offer to provide more detailed information or a demo
                
                Your goal is to guide them toward the right purchase decision without being pushy."""
            ),
            HumanMessage(
                content=state["customer_query"]
            )
        ]
    )

    return {"response": output_parser.invoke(response)}

#Node to handle unknown department classification
def unknown(state: State):
    """Generate a general response when classification confidence is low"""

    response = llm.invoke(
        [
            SystemMessage(
                content="""You're Taylor from Customer Support, the first point of contact for all inquiries.
                
                This customer message could belong to multiple departments. Respond in a real-time conversational manner.
                
                - Thank them for reaching out
                - Acknowledge their message in a way that shows you've read it
                - Ask 1-2 specific clarifying questions to better understand their needs
                - Assure them you'll connect them with the right specialist once you understand their request better
                
                Make your response warm and helpful, as if you're having a real conversation rather than sending an automated message."""
            ),
            HumanMessage(
                content=state["customer_query"]
            )
        ]
    )

    return {"response": output_parser.invoke(response)}

#State graph to define the workflow
builder = StateGraph(State)

#Add Nodes
builder.add_node("Classify_Customer_Query", classify_customer_query)
builder.add_node("Billing", billing)
builder.add_node("Tech_Support", techsupport)
builder.add_node("Sales", sales)
builder.add_node("Unknown", unknown)

#Connect Nodes with Edges
builder.add_edge(START, "Classify_Customer_Query")
builder.add_conditional_edges("Classify_Customer_Query", route_condition, 
                              {"billing": "Billing", "tech_support": "Tech_Support", "sales": "Sales", "unknown": "Unknown"})
builder.add_edge("Billing", END)
builder.add_edge("Tech_Support", END)
builder.add_edge("Sales", END)
builder.add_edge("Unknown", END)

#Compile the state graph
graph = builder.compile()

# 🎨 Streamlit UI - Frontend
st.set_page_config(page_title="📞 Customer Support Query Classifier", layout="wide")

# Sidebar with workflow diagram
with st.sidebar:
    st.subheader("Workflow Diagram")

    try:
        #Generate Mermaid Workflow Diagram
        mermaid_diagram = graph.get_graph().draw_mermaid_png()

        #Save and Display the Image in the sidebar
        image_path = "workflow_diagram.png"

        with open(image_path, "wb") as f:
            f.write(mermaid_diagram)
        
        st.image(image_path, caption="Workflow Execution", use_container_width=True)
    except Exception as e:
        st.error(f"Unable to generate workflow diagram: {e}")
        st.info("The workflow still functions correctly even without the visualization.")


# Define sample questions by category
billing_samples = [
    "I was charged twice for my monthly subscription",
    "How do I update my credit card information?",
    "When will my refund be processed?",
    "I need a copy of my last invoice",
    "Can I change my billing cycle from monthly to annual?"
]

tech_support_samples = [
    "The app keeps crashing when I try to upload photos",
    "I forgot my password and can't reset it",
    "The dashboard isn't showing my latest data",
    "I'm getting an error code XZ-404 when I try to login",
    "How do I connect your software to my email account?"
]

sales_samples = [
    "What's the difference between your Basic and Pro plans?",
    "Do you offer discounts for educational institutions?",
    "I want to upgrade my subscription to the enterprise level",
    "Does your product support integration with Salesforce?",
    "Can I get a demo of your new features?"
]

ambiguous_samples = [
    "I need help with my account",
    "I'm having problems with your product",
    "Can someone please contact me as soon as possible?",
    "What are the steps to set this up and how much does it cost?",
    "I'm not sure if I'm in the right place"
]

# Sidebar with sample questions
st.sidebar.title("📝 Sample Questions")
st.sidebar.info("Click on any sample question to test the classifier")

st.sidebar.subheader("💰 Billing Questions")
for sample in billing_samples:
    if st.sidebar.button(sample, key=f"billing_{billing_samples.index(sample)}"):
        st.session_state.user_query = sample

st.sidebar.subheader("🔧 Tech Support Questions")
for sample in tech_support_samples:
    if st.sidebar.button(sample, key=f"tech_{tech_support_samples.index(sample)}"):
        st.session_state.user_query = sample

st.sidebar.subheader("🛒 Sales Questions")
for sample in sales_samples:
    if st.sidebar.button(sample, key=f"sales_{sales_samples.index(sample)}"):
        st.session_state.user_query = sample

st.sidebar.subheader("❓ Ambiguous Questions")
for sample in ambiguous_samples:
    if st.sidebar.button(sample, key=f"ambig_{ambiguous_samples.index(sample)}"):
        st.session_state.user_query = sample

# Initialize session state for user query if it doesn't exist
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

# Main content area
st.title("📞 AI-Powered Customer Support Query Classifier")
st.write("This system classifies customer queries into **Billing, Tech Support, or Sales** departments.")

# 📍 User input field - show the selected sample or allow manual input
user_query = st.text_area("📝 Enter your customer query:", value=st.session_state.user_query, height=100)

# Clear button
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🧹 Clear"):
        st.session_state.user_query = ""
        user_query = ""
        st.experimental_rerun()

# 🔎 Query Classification Button
if st.button("🔎 Classify Query") or (user_query != "" and user_query != st.session_state.get('last_processed_query', "")):
    if not user_query:
        st.warning("⚠️ Please enter a query before proceeding.")
    else:
        # Store the last processed query to prevent duplicate processing
        st.session_state.last_processed_query = user_query
        
        with st.spinner("🔄 Analyzing query..."):
            # 🚀 Invoke LangGraph Workflow
            state = graph.invoke({"customer_query": user_query})
        
        # Create two columns for response and classification details
        response_col, info_col = st.columns([3, 1])
        
        with response_col:
            st.subheader("💬 Response:")
            st.markdown(f"**{state['response']}**")
        
        with info_col:
            st.subheader("🏷️ Classification:")
            
            # Display department with appropriate emoji
            dept_emoji = {"billing": "💰", "tech_support": "🔧", "sales": "🛒"}.get(state['department'], "❓")
            st.markdown(f"**Department:** {dept_emoji} {state['department'].upper()}")
            
            # Display confidence with color coding
            confidence = state['confidence']
            if confidence >= 0.8:
                st.markdown(f"**Confidence:** 🟢 {confidence:.2f}")
            elif confidence >= 0.7:
                st.markdown(f"**Confidence:** 🟡 {confidence:.2f}")
            else:
                st.markdown(f"**Confidence:** 🔴 {confidence:.2f}")
            
            st.markdown(f"**Reason:** {state['reason']}")

st.markdown("---")
st.caption("Powered by **LangGraph Routing Workflow, Groq and Streamlit** 🚀")