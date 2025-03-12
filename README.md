# Story Extender

A web application that extends existing stories using Retrieval-Augmented Generation (RAG). 

## Overview

Story Extender allows users to upload a text file containing a story and then generates extensions or continuations to that story. The application uses RAG techniques to generate contextually relevant and coherent extensions.

## Features

- Upload text files containing stories
- Adjust the length of the story extension
- View both the original story and the extension side by side
- Contextually relevant extensions using RAG

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```

## Usage

1. Upload a text file containing your story
2. Adjust the desired length of the extension using the slider
3. Click the "Extend Story" button
4. View the extended story in the output area

## Technical Details

The application uses:
- Gradio for the web interface
- RAG (Retrieval-Augmented Generation) for generating story extensions
- LangChain for orchestrating the RAG pipeline

## Environment Variables

Create a `.env` file with the following variables:
- `OPENAI_API_KEY`: Your OpenAI API key

## Future Improvements

- Add support for different language models
- Implement different extension styles
- Allow for controlling tone, style, and genre of extensions
- Add persistence to save stories and their extensions 