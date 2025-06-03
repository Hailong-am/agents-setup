import os

import boto3
from dotenv import load_dotenv

load_dotenv()

def get_root_session():
    session = boto3.Session(profile_name=os.environ['AWS_PROFILE'])
    sts = session.client('sts')
    assumed_role = sts.assume_role(
        RoleArn=os.environ['AWS_ASSUME_ROLE'],
        RoleSessionName="AssumeRoleSession",
        DurationSeconds=3600
    )
    creds = assumed_role['Credentials']
    root_session = boto3.Session(
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'],
    )
    return root_session