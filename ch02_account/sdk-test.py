import boto3
from pprint import pprint
boto3.setup_default_session(profile_name="awsstudent")
s3 = boto3.client('s3')
pprint(s3.list_buckets())