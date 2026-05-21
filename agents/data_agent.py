import os
import ast
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── Initialize the LLM ────────────────────────────────────
def get_llm():
    """Initialize and return the Groq LLM."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# ── System Prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """
You are QueryQuill 🪶, an expert AI Data Analyst assistant.
You help users understand their datasets by answering questions
in a clear, friendly, and insightful way.

You have been given a dataset with the following details:
{dataset_summary}

Your capabilities:
- Answer questions about the data in plain English
- Identify trends, patterns, and anomalies
- Suggest useful visualizations
- Provide statistical insights
- Explain what the data means in simple terms

Rules:
- Always be specific — refer to actual column names and values
- If asked to plot or visualize, describe what chart would be best
- If you don't know something, say so honestly
- Keep responses concise but informative
- Use bullet points and formatting to make answers readable
"""

# ── Build Prompt Template ─────────────────────────────────
def get_prompt_template():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])

# ── Chat History Manager ──────────────────────────────────
class ChatHistoryManager:
    def __init__(self):
        self.history = []

    def add_user_message(self, message: str):
        self.history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.history.append(AIMessage(content=message))

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []

# ── Main Agent Function ───────────────────────────────────
def ask_agent(
    user_input: str,
    dataset_summary: str,
    chat_history_manager: ChatHistoryManager
) -> str:
    """
    Send a question to the AI agent and get a response.
    """
    try:
        llm = get_llm()
        prompt = get_prompt_template()

        # Build the chain
        chain = prompt | llm

        # Get response
        response = chain.invoke({
            "dataset_summary": dataset_summary,
            "chat_history": chat_history_manager.get_history(),
            "user_input": user_input
        })

        ai_response = response.content

        # Save to history
        chat_history_manager.add_user_message(user_input)
        chat_history_manager.add_ai_message(ai_response)

        return ai_response

    except Exception as e:
        return f"❌ Agent error: {str(e)}"


# ── Suggested Questions Generator ────────────────────────
def generate_suggested_questions(dataset_summary: str) -> list:
    """Generate smart suggested questions based on the dataset."""
    try:
        llm = get_llm()

        prompt = f"""
Based on this dataset summary, generate exactly 4 short,
interesting questions a user might ask about this data.
Return ONLY a Python list of 4 strings, nothing else.
Example format: ["Question 1?", "Question 2?", "Question 3?", "Question 4?"]

Dataset Summary:
{dataset_summary}
        """

        response = llm.invoke(prompt)

        # Safely parse the list
        clean = response.content.strip().replace("```python", "").replace("```", "")
        questions = ast.literal_eval(clean)
        return questions

    except Exception:
        return [
            "What are the main trends in this dataset?",
            "Which columns have missing values?",
            "What is the distribution of numeric columns?",
            "Can you summarize this dataset for me?"
        ]