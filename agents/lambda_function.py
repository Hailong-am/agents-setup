import base64
import os

import boto3
import json
import logging
from urllib.parse import urlencode

import requests

from agents.tokens import get_root_session

logger = logging.getLogger()
logger.setLevel(logging.INFO)

roleArn = os.environ['ROLE_ARN']
t2ppl_roleArn = os.environ['T2PPL_ROLE_ARN']

root_session = boto3.Session()
# root_session = get_root_session()


def decrypt_kms(encrypted_text: str):
    kms_client = root_session.client('kms')
    decrypted_text = kms_client.decrypt(
        CiphertextBlob=base64.b64decode(encrypted_text)
    )['Plaintext'].decode('utf-8')
    return decrypted_text


def lambda_handler(event, context):
    refresh_datasource_title = event['refresh_datasource_title']
    broken_datasource_ids = event['broken_datasource_ids']
    play_ground_host = event['playground_host']
    auth_b64 = event['credential']
    refresh_models_ids = event['refresh_models_ids']

    decrypted_auth = decrypt_kms(auth_b64)

    credentials = get_temp_credentials(roleArn)
    t2ppl_credentials = get_temp_credentials(t2ppl_roleArn)
    client = PlaygroundClient(play_ground_host, datasource_id=None, auth=decrypted_auth)
    ds_list = client.query_all_datasource()
    # append an empty datasource for local cluster
    ds_list.append({'id': '', 'title': 'Local Cluster', 'endpoint': 'localhost'})

    for ds in ds_list:
        logger.info(f"start process datasource: {ds['id']} - {ds['title']} - {ds['endpoint']}")
        if is_aos(ds['endpoint']):
            continue
        if ds['id'] in [broken_datasource_ids]:
            continue
        if len(refresh_datasource_title) > 0 and ds['title'] not in refresh_datasource_title:
            continue
        client.set_datasource_id(ds['id'])
        models = client.query_models()
        # a list given by user
        if len(refresh_models_ids) > 0:
            models = list(filter(lambda x: x['model_id'] in refresh_models_ids, models))

        # extract model ids
        model_ids = list(map(lambda model: model['model_id'], models))
        client.undeploy_model(model_ids)

        update_connector_credentials(client, models, credentials, t2ppl_credentials)
        # update_model_credentials(client, models, credentials, t2ppl_credentials)

        model_sanity_check(client, model_ids)


def model_sanity_check(client, model_ids):
    # loop model_ids and predict
    for model_id in model_ids:
        logger.info(f"perform sanity test on model_id: {model_id}")
        client.deploy_model(model_id)
        predict_resp = client.predict_model(model_id)
        if 'error' in json.loads(predict_resp):
            logger.error(f"predict error: {predict_resp}")
            continue
        else:
            logger.info(f"predict success: {predict_resp}")


def update_model_credentials(client, models, credentials, t2ppl_credentials):
    embedd_models = list(filter(lambda x: x['connector'] is not None, models))
    for embedd_model in embedd_models:
        connector = embedd_model['connector']
        if 'protocol' not in connector:
            continue
        if 'credential' not in connector:
            continue
        if 'actions' not in connector:
            continue
        action = connector['actions'][0]
        url = action['url']
        if "bedrock" in url:
            client.rotate_model_token(embedd_model['model_id'], credentials)


def update_connector_credentials(client, models, credentials, t2ppl_credentials):
    connector_ids = list(map(lambda model: model['connector_id'], models))
    # filter out None values
    connector_ids = list(filter(lambda x: x is not None, connector_ids))
    for connector_id in connector_ids:
        connector_type = client.connector_type(connector_id)
        if connector_type['bedrock']:
            logger.info(f"start processing bedrock connector: {connector_id}, refresh token...")
            client.rotate_connector_token(connector_id, credentials)
        elif connector_type['sagemaker']:
            logger.info(f"start processing sagemaker connector: {connector_id}, refresh token...")
            client.rotate_connector_token(connector_id, t2ppl_credentials)


def get_temp_credentials(role: str):
    sts = root_session.client('sts')
    assumed_role = sts.assume_role(
        RoleArn=role,
        RoleSessionName="AssumeRoleSession",
        DurationSeconds=3600
    )
    credentials = assumed_role['Credentials']
    return credentials


def is_aos(endpoint: str) -> bool:
    aos = 'es.amazonaws.com' in endpoint or 'aoss.amazonaws.com' in endpoint or 'es-staging.amazonaws.com' in endpoint
    # .aos.us-east-1.on.aws
    aos_dual_stack = '.aos.' in endpoint and '.on.aws' in endpoint
    return aos or aos_dual_stack


# def encrypt_kms(plain_text: str):
#     kms_client = root_session.client('kms')
#     encrypted_text = kms_client.encrypt(
#         KeyId='arn:aws:kms:us-west-2:330700426359:key/0d58ea37-443e-4860-9ebf-7f5b2c9408ad',
#         Plaintext=plain_text
#     )['CiphertextBlob']
#     return base64.b64encode(encrypted_text).decode('utf-8')


class PlaygroundClient:
    def __init__(self, host: str, auth: str, datasource_id: str = None):
        self.host = host
        self.dev_tool_proxy = f"{host}/api/console/proxy"
        self.datasource_id = datasource_id
        self.headers = {
            "Content-type": "application/json",
            "osd-xsrf": "osd-fetch",
            "Authorization": auth
        }

    # setter for datasource_id
    def set_datasource_id(self, datasource_id: str):
        self.datasource_id = datasource_id

    def send_request(self, endpoint: str, payload: object, method="post"):
        try:
            response = requests.request(method, url=endpoint, headers=self.headers, json=payload)
            logger.debug(response.text)
            if response.status_code != 200:
                return json.dumps({"error": response.text})
            return response.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def query_all_datasource(self):
        host = f'{self.host}/api/saved_objects/_find?fields=id&fields=endpoint&fields=title&fields=dataSourceVersion&fields=dataSourceEngineType&per_page=10000&type=data-source'
        ds_response = self.send_request(host, {}, method="get")
        ds_list = json.loads(ds_response)
        # saved_objects
        saved_objects = ds_list['saved_objects']
        # convert to a dict keys is id value is title
        saved_objects = list(map(lambda saved_object: {
            "id": saved_object['id'],
            "title": saved_object['attributes']['title'],
            "endpoint": saved_object['attributes']['endpoint']
        }, saved_objects))

        return saved_objects

    def query_models(self):
        query_body = {
            "_source": ["_id", "connector_id", "model_state", "connector"],
            "size": 1000,
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "algorithm": {
                                    "value": "REMOTE"
                                }
                            }
                        },
                        {
                            "term": {
                                "model_state": {
                                    "value": "DEPLOYED"
                                }
                            }
                        }
                    ]
                }
            }
        }

        model_search_parameters = {
            "path": "/_plugins/_ml/models/_search",
            "method": "GET",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.dev_tool_proxy}?{urlencode(model_search_parameters)}"
        models_text = self.send_request(endpoint, payload=query_body)
        model_json = json.loads(models_text)
        # if model_json has error
        if 'error' in model_json or model_json["hits"]["total"] == 0:
            return []
        models = []
        for model in model_json['hits']['hits']:
            model_id = model['_id']
            connector_id = model['_source']['connector_id'] if 'connector_id' in model['_source'] else None
            connector = model['_source']['connector'] if 'connector' in model['_source'] else None
            models.append({
                "model_id": model_id,
                "connector_id": connector_id,
                "connector": connector
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

        endpoint = f"{self.dev_tool_proxy}?{urlencode(undeploy_parameters)}"

        self.send_request(endpoint, undeploy_body)

    def deploy_model(self, model_id: str):
        deploy_parameters = {
            "path": f"/_plugins/_ml/models/{model_id}/_deploy",
            "method": "POST",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.dev_tool_proxy}?{urlencode(deploy_parameters)}"

        self.send_request(endpoint, {})

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

        endpoint = f"{self.dev_tool_proxy}?{urlencode(rotate_parameters)}"

        self.send_request(endpoint, rotate_body)

    def rotate_model_token(self, model_id: str, credentials: dict[str, str]):
        rotate_body = {
            "connector": {
                "credential": {
                    "access_key": credentials['AccessKeyId'],
                    "secret_key": credentials['SecretAccessKey'],
                    "session_token": credentials['SessionToken']
                }
            }
        }

        rotate_parameters = {
            "path": f"/_plugins/_ml/models/{model_id}",
            "method": "PUT",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.dev_tool_proxy}?{urlencode(rotate_parameters)}"

        self.send_request(endpoint, rotate_body)

    def connector_type(self, connector_id: str):
        get_connector_parameters = {
            "path": f"/_plugins/_ml/connectors/{connector_id}",
            "method": "GET",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.dev_tool_proxy}?{urlencode(get_connector_parameters)}"

        response_text = self.send_request(endpoint, {})
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

    def predict_model(self, model_id: str):
        invoke_parameters = {
            "path": f"/_plugins/_ml/models/{model_id}/_predict",
            "method": "POST",
            "dataSourceId": self.datasource_id
        }

        endpoint = f"{self.dev_tool_proxy}?{urlencode(invoke_parameters)}"
        payload = {
            "parameters": {
                "prompt": "How are you",
                "inputText": "hello",
                "input": "hello"
            }
        }

        return self.send_request(endpoint, payload)
