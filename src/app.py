import gradio as gr
from src.agent import run_agent

def save_profile(job_type, income_source, notices):
    """
    Saves the user's profile based on the initial form inputs.
    """
    pass


def chat_interface(message, history, user_profile):
    """
    The main chat function that receives the user's question, 
    passes it to the run_agent pipeline, and returns the answer.
    """
    pass


def generate_next_steps(history, user_profile):
    """
    Generates an actionable summary checklist based on the chat history and profile.
    """
    pass


# We will build the Gradio Blocks UI down here later
if __name__ == "__main__":
    print("Gradio app skeleton initialized.")
