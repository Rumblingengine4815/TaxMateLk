import gradio as gr
from src.agent import run_agent_stream

# We will store the user's profile globally for this simple demo
USER_STATE = {}

def save_profile(job_type, income_source, notices):
    """
    Saves the user's profile based on the initial form inputs.
    """
    USER_STATE["job_type"] = job_type
    USER_STATE["income_source"] = income_source
    USER_STATE["notices"] = notices
    return f"Profile Saved! Job: {job_type}, Income: {income_source}"

def chat_interface(message, history, uploaded_pdf):
    """
    The main chat function that receives the user's question, 
    passes it to the run_agent_stream pipeline, and yields the answer.
    """
    for output in run_agent_stream(message, user_profile=USER_STATE, pdf_path=uploaded_pdf):
        yield output

def generate_next_steps():
    """
    Generates an actionable summary based on the profile.
    """
    if not USER_STATE:
        return "Please save your profile first!"
        
    return f"Based on your profile as a {USER_STATE.get('job_type', 'user')} earning from {USER_STATE.get('income_source', 'unknown sources')}, please ask your tax questions in the chat to get accurate calculations and advice."

# ---------------------------------------------------------
# GRADIO UI DEFINITION (The "Frontend")
# ---------------------------------------------------------
custom_css = """
.disclaimer { font-size: 0.85em; color: gray; text-align: center; margin-top: -10px; margin-bottom: 15px; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css) as demo:
    gr.Markdown("# 🇱🇰 TaxMate LK — Sri Lankan Tax Advisory Assistant")
    gr.Markdown("<div class='disclaimer'>This is AI-generated guidance based on June 2026 IRD regulations. Consult a tax professional for official advice.</div>")
    
    with gr.Row():
        # Left Column: The Profile Form & Example Questions
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Step 1: Your Profile")
            job_input = gr.Textbox(label="What is your profession?", placeholder="e.g. Photographer, Freelancer, Consultant")
            income_input = gr.Dropdown(choices=["Local Income", "Foreign Income", "Both"], value="Local Income", label="Income Source")
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
            
            pdf_upload = gr.File(
                label="Optional: Upload your IRD Notice or Tax Document",
                file_types=[".pdf"],
                type="filepath"
            )

            gr.Markdown("**Tip:** Upload a text-based PDF for best results. Scanned images are not supported.")
            
            chat = gr.ChatInterface(
                fn=chat_interface,
                additional_inputs=[pdf_upload],
                examples=[
                    ["I am a photographer. Do I have to pay withholding tax from June 2026? If so, how much on a 150000 payment?", None],
                    ["What is the exact deadline for the 2026 tax interest waiver?", None],
                    ["I earn Rs. 4,000,000 annually. Calculate my total income tax.", None],
                    ["When are the quarterly tax installments due?", None]
                ],
                title="",
                description="The agent pipeline explicitly shows its steps: Classifying -> Retrieving -> Calculating -> Answering.",
                fill_height=True
            )

if __name__ == "__main__":
    print("Launching TaxMate LK UI...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
