"""
Intro Bedrock
"""

import boto3
import pprint
import json


session = boto3.Session(profile_name="personal")
bedrock = session.client(
    service_name="bedrock",
    region_name="us-east-1",
)
bedrock_runtime = session.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

pp = pprint.PrettyPrinter(indent=4)


def list_foundation_models():
    """
    List foundation models available in Bedrock.
    """
    response = bedrock.list_foundation_models()
    for model in response.get("modelSummaries", []):
        pp.pprint(model)


def generate(prompt: str):
    titan_model_id = "amazon.titan-text-express-v1"
    titan_config = json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                # "maxTokensCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1,
            },
        }
    )
    response = bedrock_runtime.invoke_model(
        modelId=titan_model_id,
        body=titan_config,
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response.get("body").read().decode("utf-8"))
    return response_body


if __name__ == "__main__":
    print("Hello, Bedrock!")
    # list_foundation_models()
    prompt = "tell me a story about a dragon. Keep it short (3 paragraphs)."
    response = generate(prompt)
    print("Response: ", response)
