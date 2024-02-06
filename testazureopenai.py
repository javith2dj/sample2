import os
import openai

# Set OpenAI configuration settings
openai.api_type = "azure"
openai.api_base = "https://cog-2iwhormj3dgc4.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "3d21de1940a849b3bd4c97c710e35f2b"

# Send request to Azure OpenAI model
print("Sending request for summary to Azure OpenAI endpoint...\n\n")
response = openai.ChatCompletion.create(
    engine="chat",
    temperature=0.7,
    max_tokens=120,
    messages=[
       {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Where is India"}
    ]
)

print("Summary: " + response.choices[0].message.content + "\n")