import os
import time
import logging
import gradio as gr
import whisper
from pydub import AudioSegment
from openai import OpenAI
from lumaai import LumaAI, APIConnectionError, APIStatusError

# Configure logging
logging.basicConfig(
    filename="transcription_log.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set API keys
LUMA_API_KEY = "your_luma_api_key_here"  # Replace with your actual Luma AI API key
OPENAI_API_KEY = "your_openai_api_key_here"  # Replace with your actual OpenAI API key

# Initialize API clients
luma_client = LumaAI(auth_token=LUMA_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Load Whisper model
model = whisper.load_model("base")


# Helper function to format timestamps for SRT, VTT, and text
def format_timestamp(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:02},{milliseconds:03}"


# Function to write structured text transcript with timestamps
def write_structured_text(segments):
    transcript = ""
    for segment in segments:
        start_time = format_timestamp(segment["start"])
        text = segment["text"].strip()
        transcript += f"[{start_time}] {text}\n\n"
    return transcript


# Transcription function
def transcribe(audio_file):
    try:
        logging.info("Received an audio file for transcription.")
        audio_path = audio_file.name

        # Convert audio to WAV using pydub
        wav_path = "converted_audio.wav"
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
        logging.info("Audio file converted to WAV.")

        # Transcribe using Whisper
        result = model.transcribe(wav_path, language="en")
        structured_text = write_structured_text(result["segments"])

        logging.info("Transcription completed successfully.")
        return structured_text

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return str(e)


# Thematic Analysis with OpenAI
def thematic_analysis(transcript):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user",
                       "content": f"Please provide a thematic analysis of the following transcript:\n{transcript}"}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error during thematic analysis: {str(e)}")
        return "Error during thematic analysis."


# Overall Insights with OpenAI
def overall_insights(transcript):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Provide overall insights from the following transcript:\n{transcript}"}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error during overall insights generation: {str(e)}")
        return "Error during overall insights generation."


# Function to refine the prompt using OpenAI
def refine_prompt(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": (
                "Refine the following prompt for a video generation AI based on these guidelines:\n\n"
                "- Be specific in describing the main subject and setting.\n"
                "- Include important details.\n"
                "- Focus on emotions or atmosphere.\n"
                "- Use simple language."
            )}, {"role": "user", "content": f"User Prompt: '{prompt}'\n\nRefined Prompt:"}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error refining prompt: {str(e)}")
        return f"Error refining prompt: {str(e)}"


# Function to generate video using Luma AI
def generate_video(prompt):
    refined_prompt = refine_prompt(prompt)
    if "Error" in refined_prompt:
        return refined_prompt, None

    try:
        generation = luma_client.generations.create(
            prompt=refined_prompt,
            aspect_ratio="16:9",
            loop=False
        )

        generation_id = generation.id
        completed = False
        while not completed:
            generation = luma_client.generations.get(id=generation_id)
            if generation.state == "completed":
                return refined_prompt, generation.assets.video
            elif generation.state == "failed":
                return f"Generation failed: {generation.failure_reason}", None
            time.sleep(3)

    except (APIConnectionError, APIStatusError) as e:
        return f"Error in video generation: {str(e)}", None


# Define Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## Multi-Function App: Transcription and Video Generation")

    # Transcription Tab
    with gr.Tab("Transcription"):
        with gr.Row():
            audio_input = gr.File(label="Upload Audio File", type="filepath")
            transcribe_button = gr.Button("Transcribe")
        transcript_output = gr.Textbox(label="Diarized Transcript", lines=10, interactive=False)

        # Thematic Analysis and Overall Insights Buttons
        analysis_button = gr.Button("Thematic Analysis")
        insights_button = gr.Button("Overall Insights")

        thematic_analysis_output = gr.Textbox(label="Thematic Analysis Result", lines=5)
        overall_insights_output = gr.Textbox(label="Overall Insights Result", lines=5)

        # Bind Functions
        transcribe_button.click(
            fn=transcribe,
            inputs=audio_input,
            outputs=transcript_output
        )
        analysis_button.click(
            fn=thematic_analysis,
            inputs=transcript_output,
            outputs=thematic_analysis_output
        )
        insights_button.click(
            fn=overall_insights,
            inputs=transcript_output,
            outputs=overall_insights_output
        )

    # Video Generation Tab
    with gr.Tab("Video Generation"):
        prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Describe your scene...")
        generate_button = gr.Button("Generate Video")
        refined_prompt_display = gr.Textbox(label="Refined Prompt")
        video_output = gr.Video(label="Generated Video")

        generate_button.click(
            fn=generate_video,
            inputs=prompt_input,
            outputs=[refined_prompt_display, video_output]
        )

# Launch Gradio app
app.launch()

