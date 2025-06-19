"""
Embedding generation intro.
"""

import boto3
import json


session = boto3.Session(profile_name="personal")
client = session.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

fact = "The first moon landing was in 1969"
animal = "cat"

response = client.invoke_model(
    modelId="amazon.titan-embed-text-v1",
    body=json.dumps(
        {
            "inputText": fact,
        }
    ),
    contentType="application/json",
    accept="application/json",
)

response_body = json.loads(response.get("body").read().decode("utf-8"))
print(response_body.get("embedding", []))
