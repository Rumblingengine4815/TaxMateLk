import os

import gradio as gr

from src.agent import (
    extract_user_pdf,
    step_1_classify,
    step_2_retrieve,
    step_3_calculate,
    step_4_generate,
)

PROFILE_STATE: dict = {}

EXAMPLES = [
    "I am a photographer earning Rs 150,000 per month. How much WHT will my client deduct?",
    "What is the exact deadline for the 2026 interest waiver?",
    "Can a senior citizen still file on paper in 2025/2026?",
    "When are the quarterly tax installments due?",
]


def save_profile(job_type, income_source, notices):
    PROFILE_STATE["job_type"] = job_type or ""
    PROFILE_STATE["income_source"] = income_source or ""
    PROFILE_STATE["notices"] = bool(notices)
    return f"Saved: {job_type} | {income_source} | notice={bool(notices)}"


def next_steps():
    if not PROFILE_STATE:
        return "Save your profile first, then ask a tax question."
    return (
        f"Use the chat to ask about {PROFILE_STATE.get('job_type', 'your work')} "
        f"and {PROFILE_STATE.get('income_source', 'your income source')}. "
        "The app will show retrieved IRD sources and the full step trace."
    )


def respond(message, history, uploaded_pdf):
    if not message.strip():
        yield history, "Type a question to start.", ""
        return

    history = history + [(message, "Classifying...")]
    yield history, "Classifying...", ""

    try:
        category = step_1_classify(message)
        history[-1] = (message, f"Classified as {category}. Retrieving official IRD documents...")
        yield history, "Retrieving...", ""

        context, sources = step_2_retrieve(message)
        if uploaded_pdf:
            uploaded_text = extract_user_pdf(uploaded_pdf)
            context = f"{context}\n\n[USER UPLOADED DOCUMENT]\n{uploaded_text[:3000]}"

        history[-1] = (
            message,
            f"Classified as {category}. Retrieved: {', '.join(sources) if sources else 'no indexed docs'}. Calculating..."
        )
        yield history, "Calculating...", "\n".join(sources) if sources else "No document retrieved"

        tool_output = step_3_calculate(category, message)

        history[-1] = (message, "Calculating done. Generating cited answer...")
        yield history, "Answering...", "\n".join(sources) if sources else "No document retrieved"

        answer = step_4_generate(message, context, tool_output, PROFILE_STATE or None, sources=sources)
        history[-1] = (message, answer)
        trace = [
            f"Step 1: {category}",
            f"Step 2 sources: {', '.join(sources) if sources else 'none'}",
            f"Step 3 tool output: {tool_output}",
            "Step 4: answer generated",
        ]
        yield history, "Done.", "\n".join(trace)
    except Exception as exc:
        history[-1] = (message, f"Sorry, I hit an error while answering that. {exc}")
        yield history, "Error.", "The pipeline failed safely."


custom_css = """
.disclaimer {
    font-size: 0.86rem;
    color: #666;
    text-align: center;
    margin: 0.75rem 0 1rem;
    padding: 0.45rem 0.75rem;
    line-height: 1.4;
    border-radius: 0.5rem;
    background: rgba(255, 255, 255, 0.7);
}
.hint { color: #444; font-size: 0.95rem; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css) as demo:
    gr.Markdown("# TaxMate LK - Sri Lankan Tax Advisory Assistant")
    gr.Markdown(
        "<div class='disclaimer'>This is AI-generated guidance. Consult a tax professional for official advice.</div>"
    )

    with gr.Row():
        example_buttons = []
        with gr.Column(scale=1):
            gr.Markdown("### Profile")
            job_input = gr.Textbox(label="Profession", placeholder="Photographer, freelancer, designer...")
            income_input = gr.Dropdown(
                choices=["Local Income", "Foreign Income", "Both"],
                value="Local Income",
                label="Income source",
            )
            notice_input = gr.Checkbox(label="Received an IRD notice?")
            save_btn = gr.Button("Save profile", variant="primary")
            profile_status = gr.Textbox(label="Status", interactive=False)
            next_btn = gr.Button("What should I do next?")
            next_output = gr.Textbox(label="Next steps", interactive=False, lines=4)
            save_btn.click(save_profile, [job_input, income_input, notice_input], profile_status)
            next_btn.click(next_steps, outputs=next_output)

            gr.Markdown("### Example questions")
            for example in EXAMPLES:
                example_buttons.append(gr.Button(example))
            gr.Markdown("<div class='hint'>Click an example, paste it into chat, or type your own question.</div>")

        with gr.Column(scale=2):
            gr.Markdown("### Chat")
            pdf_upload = gr.File(
                label="Optional IRD notice or PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            chatbot = gr.Chatbot(label="Conversation", height=520, type="tuples")
            step_status = gr.Textbox(label="Live step", interactive=False)
            evidence = gr.Textbox(label="Retrieved sources and trace", interactive=False, lines=8)
            with gr.Row(equal_height=True):
                message = gr.Textbox(
                    label="Ask a tax question",
                    placeholder="Type here and press Enter",
                    scale=5,
                )
                send = gr.Button("Send", variant="primary", scale=1)

            message.submit(
                respond,
                inputs=[message, chatbot, pdf_upload],
                outputs=[chatbot, step_status, evidence],
            )
            send.click(
                respond,
                inputs=[message, chatbot, pdf_upload],
                outputs=[chatbot, step_status, evidence],
            )

    for button, example in zip(example_buttons, EXAMPLES):
        button.click(lambda q=example: q, outputs=message)

def launch():
    return demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )


if __name__ == "__main__":
    print("Launching TaxMate LK UI...")
    launch()
