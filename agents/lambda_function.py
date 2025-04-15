import base64
import os

import boto3
import json
import logging
from urllib.parse import urlencode

import requests

logger = logging.getLogger()
logger.setLevel(logging.INFO)

roleArn = os.environ['ROLE_ARN']
t2ppl_roleArn = os.environ['T2PPL_ROLE_ARN']
play_ground_host = os.environ['PLAYGROUND_HOST']
dev_tool_proxy = f"{play_ground_host}/api/console/proxy"
auth_b64 = os.environ['CREDENTIAL']

root_session = boto3.Session()

def decrypt_kms(encrypted_text: str):
    kms_client = root_session.client('kms')
    decrypted_text = kms_client.decrypt(
        CiphertextBlob=base64.b64decode(encrypted_text)
    )['Plaintext'].decode('utf-8')
    return decrypted_text

headers = {
    "Content-type": "application/json",
    "osd-xsrf": "osd-fetch",
    "Authorization": decrypt_kms(auth_b64)
}

def lambda_handler(event, context):
    refresh_datasource_title = event['refresh_datasource_title']
    broken_datasource_ids = event['broken_datasource_ids']
    credentials = get_temp_credentials(roleArn)
    t2ppl_credentials = get_temp_credentials(t2ppl_roleArn)
    ds_list = PlaygroundClient.query_all_datasource()

    for ds in ds_list:
        if ds['id'] in [broken_datasource_ids]:
            continue
        if len(refresh_datasource_title) > 0 and ds['title'] not in refresh_datasource_title:
            continue
        logger.info(f"start process datasource: {ds['id']} - {ds['title']}")
        model_api = PlaygroundClient(host=dev_tool_proxy, datasource_id=ds['id'])
        models = model_api.query_models()

        # extract model ids
        model_ids = list(map(lambda model: model['model_id'], models))
        model_api.undeploy_model(model_ids)

        connector_ids = list(map(lambda model: model['connector_id'], models))
        # loop connectors and get connector id
        for connector_id in connector_ids:
            connector_type = model_api.connector_type(connector_id)
            if connector_type['bedrock']:
                logger.info(f"start processing bedrock connector: {connector_id}, refresh token...")
                model_api.rotate_connector_token(connector_id, credentials)
            elif connector_type['sagemaker']:
                logger.info(f"start processing sagemaker connector: {connector_id}, refresh token...")
                model_api.rotate_connector_token(connector_id, t2ppl_credentials)

        # loop model_ids and predict
        for model_id in model_ids:
            logger.info(f"perform sanity test on model_id: {model_id}")
            predict_resp = model_api.predict_bedrock_model(model_id)
            if 'error' in json.loads(predict_resp):
                logger.error(f"predict error: {predict_resp}")
                continue
            else:
                logger.info(f"predict success: {predict_resp}")


def get_temp_credentials(role: str):
    sts = root_session.client('sts')
    assumed_role = sts.assume_role(
        RoleArn=role,
        RoleSessionName="AssumeRoleSession",
        DurationSeconds=3600
    )
    credentials = assumed_role['Credentials']
    return credentials

def decrypt_kms(encrypted_text: str):
    kms_client = root_session.client('kms')
    decrypted_text = kms_client.decrypt(
        CiphertextBlob=base64.b64decode(encrypted_text)
    )['Plaintext'].decode('utf-8')
    return decrypted_text

def send_request(endpoint: str, payload: object, method="post"):
    try:
        response = requests.request(method, url=endpoint, headers=headers, json=payload)
        logger.debug(response.text)
        if response.status_code != 200:
            return json.dumps({"error": response.text})
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


class PlaygroundClient:
    def __init__(self, host: str, datasource_id: str):
        self.host = host
        self.datasource_id = datasource_id

    @staticmethod
    def query_all_datasource():
        host = f'{play_ground_host}/api/saved_objects/_find?fields=id&fields=description&fields=title&fields=dataSourceVersion&fields=dataSourceEngineType&fields=installedPlugins&per_page=10000&type=data-source'
        ds_response = send_request(host, {}, method="get")
        ds_list = json.loads(ds_response)
        # saved_objects
        saved_objects = ds_list['saved_objects']
        # convert to a dict keys is id value is title
        saved_objects = list(map(lambda saved_object: {
            "id": saved_object['id'],
            "title": saved_object['attributes']['title']
        }, saved_objects))

        return saved_objects

    def query_models(self):
        query_body = {
            "_source": ["_id", "connector_id", "model_state"],
            "size": 1000,
            "query": {
                "term": {
                    "algorithm": {
                        "value": "REMOTE"
                    }
                }
            }
        }

        model_search_parameters = {
            "path": "/_plugins/_ml/models/_search",
            "method": "GET",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.host}?{urlencode(model_search_parameters)}"
        models_text = send_request(endpoint, payload=query_body)
        model_json = json.loads(models_text)
        # if model_json has error
        if 'error' in model_json:
            return []
        models = []
        for model in model_json['hits']['hits']:
            model_id = model['_id']
            connector_id = model['_source']['connector_id']
            models.append({
                "model_id": model_id,
                "connector_id": connector_id
            })

        return models

    def undeploy_model(self, model_ids: list[str]):
        undeploy_body = {
            "model_ids": model_ids
        }

        undeploy_parameters = {
            "path": "/_plugins/_ml/models/_undeploy",
            "method": "POST",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.host}?{urlencode(undeploy_parameters)}"

        send_request(endpoint, undeploy_body)

    def rotate_connector_token(self, connector_id: str, credentials: dict[str, str]):
        rotate_body = {
            "credential": {
                "access_key": credentials['AccessKeyId'],
                "secret_key": credentials['SecretAccessKey'],
                "session_token": credentials['SessionToken']
            }
        }

        rotate_parameters = {
            "path": f"/_plugins/_ml/connectors/{connector_id}",
            "method": "PUT",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.host}?{urlencode(rotate_parameters)}"

        send_request(endpoint, rotate_body)

    def connector_type(self, connector_id: str):
        get_connector_parameters = {
            "path": f"/_plugins/_ml/connectors/{connector_id}",
            "method": "GET",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.host}?{urlencode(get_connector_parameters)}"

        response_text = send_request(endpoint, {})
        r_json = json.loads(response_text)
        if 'error' in r_json:
            return {
                "bedrock": False,
                "sagemaker": False
            }
        url: str = r_json['actions'][0]['url']
        logger.info(f"connector url: {url}")
        return {
            "bedrock": "bedrock" in url,
            "sagemaker": "sagemaker" in url,
        }

    def predict_bedrock_model(self, model_id: str):
        invoke_parameters = {
            "path": f"/_plugins/_ml/models/{model_id}/_predict",
            "method": "POST",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.host}?{urlencode(invoke_parameters)}"
        payload = {
            "parameters": {
                "prompt": "how are you?"
            }
        }

        return send_request(endpoint, payload)

