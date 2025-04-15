import json

from dotenv import load_dotenv
import os

load_dotenv()

import requests

host = "https://localhost:9200"
headers = {
    "Content-Type": "application/json"
}
auth = (os.getenv("USERNAME"), os.getenv("PASSWORD"))

agents = [
    "/_plugins/_ml/config/os_index_type_detect",
    "/_plugins/_ml/config/os_summary",
    "/_plugins/_ml/config/os_summary_with_log_pattern",
    "/_plugins/_ml/config/os_suggest_ad",
    "/_plugins/_ml/config/os_text2vega",
    "/_plugins/_ml/config/os_text2vega_with_instructions",
    "/_plugins/_ml/config/os_data2summary",
    "/_plugins/_ml/config/os_query_assist_ppl",
]

payloads = [
    {
        "parameters": {
            "sampleData": "logs",
            "schema": ""
        }
    },
    {
        "parameters": {
            "question": "how many 404 errors last week?",
            "context": ""
        }
    },
    {
        "parameters": {
            "question": "how many 404 errors last week?",
            "context": "",
            "topNLogPatternData": ""
        }
    },
    {
        "parameters": {
            "index": "opensearch_dashboards_sample_data_ecommerce",
            "question": "how many 404 errors last week?",
            "context": "",
            "topNLogPatternData": ""
        }
    },
    {
        "parameters": {
            "index": "opensearch_dashboards_sample_data_ecommerce",
            "question": "how many 404 errors last week?",
            "context": "",
            "topNLogPatternData": "",
            "sampleData": "",
            "schema": "",
            "ppl": "",
            "input_question": "",
            "dataSchema": ""
        }
    },
    {
        "parameters": {
            "index": "opensearch_dashboards_sample_data_ecommerce",
            "question": "how many 404 errors last week?",
            "context": "",
            "topNLogPatternData": "",
            "sampleData": "",
            "schema": "",
            "ppl": "",
            "input_question": "",
            "dataSchema": "",
            "input_instruction": "bar chart"
        }
    },
    {
        "parameters": {
            "sample_count": 10,
            "total_count": 100,
            "question": "how many 404 errors last week?",
            "ppl": "",
            "sample_data": ""
        }
    },
    {
        "parameters": {
            "index": "opensearch_dashboards_sample_data_ecommerce",
            "question": "how many 404 errors last week?",
        }
    }
]

agent_execute = "/_plugins/_ml/agents/{agent_id}/_execute"

index = 0
for agent in agents:
    print(f"====={index} = {agent}====")
    url = f"{host}{agent}"
    print(url)
    # execute above url and get result
    response = requests.get(url=url, headers=headers, auth=auth)
    print(response.text)
    # {"type": "os_olly_agent", "configuration": {"agent_id": "MI5-x5UB45eVFXptC2Sp"}}
    # parse response.text as json
    agent_id = json.loads(str(response.text))['configuration']['agent_id']

    execute_url = f"{host}{agent_execute}".replace("{agent_id}", agent_id)
    response = requests.post(url=execute_url, headers=headers, auth=auth, json=payloads[index])
    print("agent execute response: ", response.text)

    index = index + 1
