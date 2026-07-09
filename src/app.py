import gradio as gr
from src.agent import run_agent

# We will store the user's profile globally for this simple demo
# In a real app like we discussed (Supabase), this goes in a database!
USER_STATE = {}

def save_profile(job_type, income_source, notices):
    """
    Saves the user's profile based on the initial form inputs.
    """
    USER_STATE["job_type"] = job_type
    USER_STATE["income_source"] = income_source
    USER_STATE["notices"] = notices
    return f"Profile Saved! Job: {job_type}, Income: {income_source}"


def chat_interface(message, history):
    """
    The main chat function that receives the user's question, 
    passes it to the run_agent pipeline, and returns the answer.
    """
    # We pass the globally stored profile to the agent so it has context
    answer = run_agent(message, user_profile=USER_STATE)
    return answer


def generate_next_steps():
    """
    Generates an actionable summary based on the profile.
    """
    if not USER_STATE:
        return "Please save your profile first!"
        
    return f"Based on your profile as a {USER_STATE.get('job_type', 'user')} earning from {USER_STATE.get('income_source', 'unknown sources')}, please check the chat above for specific tax calculations and deadlines."


# ---------------------------------------------------------
# GRADIO UI DEFINITION (The "Frontend")
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🇱🇰 TaxMate LK")
    gr.Markdown("Your personal AI Tax Assistant. Specially trained on the latest IRD regulations.")
    
    with gr.Row():
        # Left Column: The Profile Form
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Step 1: Your Profile")
            job_input = gr.Textbox(label="What is your profession? (e.g. Photographer, Freelancer)")
            income_input = gr.Dropdown(choices=["Local Income", "Foreign Income", "Both"], label="Income Source")
            notice_input = gr.Checkbox(label="Did you receive an IRD warning notice?")
            
            save_btn = gr.Button("Save Profile", variant="primary")
            profile_status = gr.Textbox(label="Status", interactive=False)
            
            save_btn.click(fn=save_profile, inputs=[job_input, income_input, notice_input], outputs=profile_status)
            
            gr.Markdown("### 📋 Next Steps")
            summary_btn = gr.Button("Generate Actionable Summary")
            summary_output = gr.Textbox(label="What you should do next:", interactive=False, lines=4)
            summary_btn.click(fn=generate_next_steps, inputs=[], outputs=summary_output)
            
        # Right Column: The Chat Interface
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Step 2: Ask Tax Questions")
            # Gradio's ChatInterface automatically handles the conversation history windowing!
            chat = gr.ChatInterface(fn=chat_interface)

if __name__ == "__main__":
    print("Launching TaxMate LK UI...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
