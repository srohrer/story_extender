import gradio as gr
import os
import json
from dotenv import load_dotenv
from vectorizer import vectorize_text, get_rag_context  # Import vectorizer functions
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Get API key from environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Global variable to store vectorized data
vectorized_data = None

def read_text_file(file):
    """Read the content of an uploaded text file."""
    if file is None:
        return None
    
    try:
        # For newer Gradio versions (NamedString objects)
        if hasattr(file, "name"):
            # Read the content from the file path
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        
        # For older Gradio versions
        if hasattr(file, "decode"):
            content = file.decode("utf-8")
            return content
        
        # If it's a path string
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            return content
            
        return f"Unsupported file type: {type(file).__name__}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def display_last_chars(text, char_limit=500):
    """Display only the last N characters of the text."""
    if text is None or text == "":
        return ""
    
    if len(text) <= char_limit:
        return text
    
    return "..." + text[-char_limit:]

def preprocess_story(text_content):
    """Vectorize the story and store the results."""
    global vectorized_data
    
    if not text_content:
        return "Please upload a story first."
    
    # Vectorize the text
    vectorized_data = vectorize_text(text_content)
    
    # Show last part of original text
    truncated_text = display_last_chars(text_content)
    
    return truncated_text, f"Story processed successfully! Created {len(vectorized_data['chunks'])} chunks."

def generate_story_extension(story_text, context, extension_length=100, tone="neutral", creativity=0.7, custom_guidance=""):
    """
    Generate a story extension using DeepSeek via langchain's ChatOpenAI interface.
    
    Args:
        story_text: The original story text
        context: The RAG context to help with the extension
        extension_length: Approximate target length in words
        tone: Tone/style for the extension (e.g., funny, serious, mysterious)
        creativity: Temperature setting for generation creativity (0.0-1.0)
        custom_guidance: User-provided guidance for the story direction
        
    Returns:
        The generated story extension
    """
    if not DEEPSEEK_API_KEY:
        return "\n\n[Error: DEEPSEEK_API_KEY not found in environment variables]"
    
    # Extract the last few paragraphs to provide as immediate context
    text_content = story_text
    last_part = text_content[-500:] if len(text_content) > 500 else text_content
    
    # Adjust tone instructions based on selected tone
    tone_instructions = ""
    if tone == "funny":
        tone_instructions = "Write in a humorous and comedic style with witty observations."
    elif tone == "serious":
        tone_instructions = "Write in a serious and thoughtful style with depth and gravity."
    elif tone == "mysterious":
        tone_instructions = "Write in an enigmatic style that creates intrigue and suspense."
    elif tone == "poetic":
        tone_instructions = "Write with poetic language, rich imagery, and lyrical flow."
    elif tone == "action-packed":
        tone_instructions = "Write with dynamic pacing, excitement, and vivid action sequences."
    
    # Add custom guidance if provided
    guidance_instructions = ""
    if custom_guidance and custom_guidance.strip():
        guidance_instructions = f"\nIncorporate the following guidance for this part of the story: {custom_guidance.strip()}"
    
    # Create the prompt for the model
    prompt = f"""You are an expert story writer. Continue the following story in a coherent and engaging manner.
    
Here's the last part of the story:
{last_part}

I'm also providing some context from other parts of the story to help maintain consistency:
{context}

Please continue the story for approximately {extension_length} words. Maintain the same tone, style, and narrative voice as the original. Do not summarize or conclude the story unless it naturally leads to an ending.

{tone_instructions}{guidance_instructions}
"""
    
    try:
        # Initialize the ChatOpenAI model configured to use DeepSeek
        chat = ChatOpenAI(
            model="deepseek-reasoner",  # DeepSeek model
            temperature=creativity,  # Use the creativity parameter
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",  # DeepSeek API endpoint
            max_tokens=extension_length * 2  # Approximate words to tokens
        )
        
        # Create a message and get completion
        messages = [HumanMessage(content=prompt)]
        response = chat.invoke(messages)
        
        # Extract and return the content
        return response.content
            
    except Exception as e:
        return f"\n\n[Error generating extension: {str(e)}]"

def extend_story(text_content, extension_length=100, tone="neutral", creativity=0.7, custom_guidance=""):
    """
    Function to extend a story using RAG and DeepSeek via langchain.
    
    Args:
        text_content: The original story text
        extension_length: Approximate target length in words
        tone: Tone/style for the extension
        creativity: Temperature setting for generation creativity
        custom_guidance: User-provided guidance for the story direction
        
    Returns:
        Just the extension text (not the full story)
    """
    global vectorized_data
    
    if not text_content:
        return "Please upload a story first."
    
    if vectorized_data is None:
        return "Please preprocess the story first by clicking the 'Preprocess Story' button."
    
    # Get RAG context using the last part of the story
    last_part = text_content[-500:] if len(text_content) > 500 else text_content
    context = get_rag_context(text_content, query=last_part)
    
    # Generate the story extension using DeepSeek
    extension = generate_story_extension(text_content, context, extension_length, tone, creativity, custom_guidance)
    
    # Return just the extension
    return extension

def save_and_update(full_text, extension):
    """
    Save extension to full text, update displays, and revectorize the story.
    This ensures that future extensions have access to the full context.
    
    Args:
        full_text: The original story text
        extension: The extension to append
        
    Returns:
        Updated full_text, complete_story, and cleared extension_text
    """
    global vectorized_data
    
    if not full_text or not extension:
        return full_text, full_text, ""
    
    # Combine the texts
    combined_text = full_text + "\n\n" + extension
    
    # Re-vectorize the updated text
    vectorized_data = vectorize_text(combined_text)
    
    # Update the process status to show revectorization
    process_status_msg = f"Story updated and re-vectorized with {len(vectorized_data['chunks'])} chunks."
    
    # Return the updated values
    return combined_text, combined_text, "", process_status_msg

# Create Gradio interface
with gr.Blocks(title="Story Extender") as app:
    gr.Markdown("# Story Extender")
    gr.Markdown("Upload a text file containing a story, and the application will extend it using DeepSeek AI.")
    
    # Store full text content (hidden from user)
    full_text = gr.State("")
    current_extension = gr.State("")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Extend Story") as extend_tab:
            with gr.Row():
                # Left column - Controls
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Story File", file_types=[".txt"])
                    preprocess_button = gr.Button("Preprocess Story")
                    original_text = gr.Textbox(label="Original Story (last 500 chars)", lines=6, interactive=False)
                    process_status = gr.Textbox(label="Processing Status", lines=1, interactive=False)
                    
                    gr.Markdown("## Extension Settings")
                    extension_length = gr.Slider(minimum=50, maximum=500, value=100, step=50, 
                                                label="Extension Length (words)")
                    
                    tone_choice = gr.Radio(
                        choices=["neutral", "funny", "serious", "mysterious", "poetic", "action-packed"],
                        value="neutral",
                        label="Tone/Style of Extension"
                    )
                    
                    creativity = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.1, 
                        label="Creativity (Temperature)"
                    )
                    
                    gr.Markdown("## Story Guidance")
                    custom_guidance = gr.Textbox(
                        label="Optional - Provide guidance for what should happen next", 
                        placeholder="Example: The main character should discover a hidden treasure, but face an unexpected challenge.",
                        lines=3
                    )
                                        
                # Right column - Extension
                with gr.Column(scale=2):
                    gr.Markdown("## Generated Extension")
                    gr.Markdown("Edit the extension below if needed, then save it to your story.")
                    extension_text = gr.Textbox(label="", lines=20, interactive=True, placeholder="Your extension will appear here...")
                    
                    with gr.Row():
                        generate_button = gr.Button("Generate Extension", variant="primary")
                        save_button = gr.Button("Save Extension to Story", variant="primary")
                        view_complete_button = gr.Button("View Complete Story")
        
        with gr.TabItem("Complete Story") as complete_tab:
            with gr.Column():
                gr.Markdown("## Complete Story")
                gr.Markdown("This tab shows your complete story with all saved extensions.")
                complete_story = gr.Textbox(label="", lines=30, interactive=False)
                
                # Download section - only in this tab
                gr.Markdown("## Download Story")
                download_button = gr.Button("Download Complete Story")
                download_output = gr.File(label="")
    
    # Event handlers
    def update_full_text(file):
        """Update the full text state with uploaded file content and display truncated version."""
        content = read_text_file(file)
        if content:
            truncated = display_last_chars(content)
            return content, truncated
        return "", ""
    
    def generate_extension(text, length, tone, creativity, guidance):
        """Generate a story extension using the provided text and length."""
        extension = extend_story(text, length, tone, creativity, guidance)
        return extension, extension
    
    def update_complete_story(full_text):
        """Update the complete story display with the current full text."""
        return full_text
    
    def switch_to_complete_tab(full_text):
        """Switch to the complete story tab and update the display."""
        return full_text, 1
    
    # File handling
    file_input.change(update_full_text, inputs=file_input, outputs=[full_text, original_text])
    
    # Preprocessing
    preprocess_button.click(preprocess_story, 
                           inputs=[full_text], 
                           outputs=[original_text, process_status])
    
    # Extension generation - single button for both initial generation and regeneration
    generate_button.click(generate_extension, 
                       inputs=[full_text, extension_length, tone_choice, creativity, custom_guidance], 
                       outputs=[extension_text, current_extension])
    
    # Save extension
    save_button.click(save_and_update, 
                     inputs=[full_text, extension_text], 
                     outputs=[full_text, complete_story, extension_text, process_status])
    
    # View complete story button
    view_complete_button.click(switch_to_complete_tab, 
                               inputs=[full_text], 
                               outputs=[complete_story, tabs])
    
    # Update complete story when switching to that tab manually
    tabs.change(update_complete_story, inputs=[full_text], outputs=[complete_story])
    
    # Download functionality
    def create_download_file(text):
        """Create a downloadable file from the complete story text."""
        if not text:
            return None
        
        # Create a temporary file
        temp_file = "temp_story.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        return temp_file
    
    download_button.click(create_download_file, inputs=[complete_story], outputs=download_output)
    
    gr.Markdown("## How to Use")
    gr.Markdown("""
    1. Upload a text file containing your story
    2. Click the 'Preprocess Story' button to vectorize your story
    3. Adjust the extension settings (length, tone, creativity)
    4. Optionally provide guidance on what should happen next
    5. Click the 'Generate Extension' button
    6. Edit the extension if needed, then save it to your story
    7. Go to the 'Complete Story' tab to view or download your complete story
    
    Note: You need a DeepSeek API key stored in your .env file as DEEPSEEK_API_KEY.
    """)

# Launch the app
if __name__ == "__main__":
    app.launch() 