

import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from example.s3_archive_workflow.fakedata import DICT_R_data_v1_2
from example.s3_archive_workflow.interface import ProjResearchDataObject
import json,msgpack
import TimeSeriesDataAnalysisLib.accessor as da
import TimeSeriesDataAnalysisLib.util as dd
import sys,os,re
# s3 data archive workflow: csv,txt... -> dict object -> class object -> s3 accessor putter
# data analytics workflow: s3 accessor get -> msgpack bytes -> dict object -> class object -> do data analytics

# s3 data archive workflow:
# dict object
dictO=DICT_R_data_v1_2
# dict object -> class object
### "force=True" if interface version not match but still force parse 
### "ele_allow_any_type=True" will allow class interface has elements of Any type like: list, dict (not recommend)
researchData=ProjResearchDataObject().loads(dictO,force=True ,ele_allow_any_type=False) 
## set other info
researchData.info.term="research_data"
researchData.info.analysis_term="A"
researchData.info.provider="myprovider"

## generate new s3 target file path if need
researchData.generate_new_key()
## validate object values before dumps
researchData.validate()

## dump as msgpack string
print("researchData\n",researchData.dumps())


# class object -> s3 accessor putter
accessKeys_path='AwsAccessKeys.csv'
region_name="ap-southeast-1"
bucket_name="my-bucket"


## create s3client for each file
encryption="key_000"
s3client=da.S3_Accessor().set_client(accessKeys_path,region_name,bucket_name=bucket_name)
s3client.encryption=encryption

def put_object(format='json'):
    ### if you don't want overwrite file on s3,  check_file_exist=True
    return s3client.put_object(researchData,check_file_exist=False,format=format)
print("put_object\n",put_object())


@dd.response_keys_filter('Key')
def list_bucket_objects():
    return s3client.list_bucket_objects()
    ###same to print(dd.response_keys_filter('Key')(s3client.list_bucket_objects)() )
print("list_bucket_objects\n",list_bucket_objects())



# data analytics workflow:
# s3 accessor get -> download decrypted file -> dict object 



# s3 accessor get -> download encrypt file

download_path='D:\\download'
reg='^research_data/A/A001/.+/key_000.json$'
list_bucket=list_bucket_objects()
target_list=list(filter(lambda x: (re.match(reg, x)) ,list_bucket))
s3client.download(target_list,download_path)


# file -> json bytes -> dict object 

### get research data
file_path=os.path.join(download_path,"research_data\\A\\A001\\2019_06\\1559644604.000\\key_000.json")
dictO2=s3client.get_dict(file_path)
print("dict dictO2\n",dictO2)
# dict object -> class object
researchData2=ProjResearchDataObject().loads(dictO2,force=True)
print("researchData2.info\n",researchData2.info)
researchData2.validate()
print("dict researchData2\n", json.loads(researchData2.dumps(), encoding='utf-8'))



#upload as encrypted file
s3client.encryption="key_001"
print("key_001")
print("put_object\n",s3client.put_object(researchData,check_file_exist=False,format='msgpack'))

#download as encrypted file
reg='^research_data/A/A001/.+/key_001.msgpack$'
target_list=list(filter(lambda x: (re.match(reg, x)) ,list_bucket))
print("download")
print(target_list,download_path)
s3client.download(target_list,download_path,encrypt=True)
print("download done")

### get record information from encrypted file
file_path=os.path.join(download_path,"research_data\\A\\A001\\2019_06\\1559644604.000\\key_001.msgpack")
dictO2=s3client.get_dict(file_path)
print(dictO2)




