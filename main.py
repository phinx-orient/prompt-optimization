import os
import streamlit as st
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optionally, you can also load environment variables from Docker Compose or Google Cloud Provider
# This typically requires additional setup depending on your deployment environment.
# For example, if using Google Cloud, you might set environment variables in your cloud function or app settings.


META_PROMPT = """
Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

# Guidelines

- Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
- Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
- Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
    - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
    - Conclusion, classifications, or results should ALWAYS appear last.
- Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
   - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
- Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
- Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
- Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
- Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
- Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
    - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
    - JSON should never be wrapped in code blocks (```) unless explicitly requested.

The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

[Concise instruction describing the task - this should be the first line in the prompt, no section header]

[Additional details as needed.]

[Optional sections with headings or bullet points for detailed steps.]

# Steps [optional]

[optional: a detailed breakdown of the steps necessary to accomplish the task]

# Output Format

[Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

# Examples [optional]

[Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
[If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

# Notes [optional]

[optional: edge cases, details, and an area to call or repeat out specific important considerations]
""".strip()


# Access the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class TaskRequest(BaseModel):
    task_or_prompt: str


def generate_prompt(request: TaskRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": META_PROMPT},
                {
                    "role": "user",
                    "content": f"Task, Goal, or Current Prompt:\n{request.task_or_prompt}",
                },
            ],
        )
        return {"prompt": completion.choices[0].message.content}
    except Exception as e:
        raise Exception(f"Error generating prompt: {str(e)}")


# Streamlit app setup
st.title("Prompt Generator")

# User input for task or prompt
user_input = st.text_area("Enter your task or prompt:")

if st.button("Generate Prompt"):
    if user_input:
        request = TaskRequest(task_or_prompt=user_input)
        result = generate_prompt(request)  # Call the existing function
        st.session_state.generated_prompt = result[
            "prompt"
        ]  # Store the result in session state
    else:
        st.error("Please enter a task or prompt.")

# Display the generated prompt
if "generated_prompt" in st.session_state:
    st.subheader("Generated Prompt:")
    editable_prompt = st.text_area(
        "Edit the generated prompt:", st.session_state.generated_prompt, height=200
    )
    st.session_state.generated_prompt = (
        editable_prompt  # Update the session state with the edited prompt
    )
