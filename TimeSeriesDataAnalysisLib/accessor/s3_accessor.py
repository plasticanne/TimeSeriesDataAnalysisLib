import sys, os
import logging
import csv
import re
from boto3 import client
import typing
import TimeSeriesDataAnalysisLib.interface.untrainable_base as ib
from TimeSeriesDataAnalysisLib.util.encryption import ENCRYPTION_MAP
from botocore.exceptions import ClientError
from io import BytesIO
import msgpack,json
class S3_Accessor:
    """a client for access s3 stroage
    
    """
    def __init__(self):
        self.__ENCRYPTION:str="key_000"
    def _set_access_keys(self,csv_path:str):
        with open(csv_path, "r", encoding='utf-8') as f:
            filereader=csv.DictReader(f)
            d = {}
            for row in filereader:
                for column, value in row.items():
                    d.setdefault(column, []).append(value)
            self._direct_set_access_keys(d["Access key ID"][0],d["Secret access key"][0])
    def _direct_set_access_keys(self,id:str,secret:str):
        self.__AWS_ACCESS_KEY_ID= id
        self.__AWS_SECRET_ACCESS_KEY = secret
    
    def set_client_from_key(self,id:str,secret:str,region_name:str,bucket_name=None)->'S3_Accessor':
        """config client s3 region and bucket
        args
        ---------
        id: aws access key id

        secret: aws access secret key

        region_name:str, aws region_name

        bucket_name:str, aws bucket_name
        """
        self._direct_set_access_keys(id,secret)
        self.__AWS_REGION_NAME=region_name
        self.__AWS_STORAGE_BUCKET_NAME=bucket_name
        self.__CLIENT = client('s3',
        region_name=region_name,
        aws_access_key_id=self.__AWS_ACCESS_KEY_ID,
        aws_secret_access_key=self.__AWS_SECRET_ACCESS_KEY,
        )
        return self
    def set_client(self,csv_path:str,region_name:str,bucket_name=None)->'S3_Accessor':
        """config client s3 region and bucket
        args
        ---------
        csv_path: aws access key csv

        region_name:str, aws region_name

        bucket_name:str, aws bucket_name
        """
        self._set_access_keys(csv_path)
        self.__AWS_REGION_NAME=region_name
        self.__AWS_STORAGE_BUCKET_NAME=bucket_name
        self.__CLIENT = client('s3',
        region_name=region_name,
        aws_access_key_id=self.__AWS_ACCESS_KEY_ID,
        aws_secret_access_key=self.__AWS_SECRET_ACCESS_KEY,
        )
        return self
    @property
    def encryption(self):
        return self.__ENCRYPTION
    @encryption.setter 
    def encryption(self,value:str):
        if value in ENCRYPTION_MAP.keys():
            self.__ENCRYPTION=value
        else:
            raise ValueError("invalid value")
    def _bucket_name_check(self,bucket_name:str):
        if self.__AWS_STORAGE_BUCKET_NAME==None and bucket_name==None:
            raise Exception('bucket_name=None, please set_client(csv_path,region_name,bucket_name="name")')
        if bucket_name==None:
            bucket_name=self.__AWS_STORAGE_BUCKET_NAME
        return bucket_name
    def head_bucket(self,):
        if self.__AWS_STORAGE_BUCKET_NAME==None:
            raise Exception('bucket_name=None, please set_client(csv_path,region_name,bucket_name="name")')
        self.__CLIENT.head_bucket(Bucket=self.__AWS_STORAGE_BUCKET_NAME,)
    def list_bucket_objects(self,bucket_name:str=None)->str:
        """get objects list in bucket
        kwargs
        ---------

        bucket_name:str, aws bucket_name, if None, access the bucket in set_client() set set_client
        """
        bucket_name=self._bucket_name_check(bucket_name)
        response = self.__CLIENT.list_objects_v2(Bucket=bucket_name,)
        return response['Contents']

    def _get_object(self,target_path:str,bucket_name:str=None):
        bucket_name=self._bucket_name_check(bucket_name)
        try:
            response=self.__CLIENT.get_object(Bucket=bucket_name,Key=target_path,
            #ResponseCacheControl='no-cache'
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise NameError('{0} dose not exist'.format(target_path))
            else:
                raise
        return response
        
    def _put_file(self,source_path:str,target_path:str,bucket_name:str=None):
        bucket_name=self._bucket_name_check(bucket_name)
        
        target_path=re.sub(r'^[\/\\]+', '', target_path)
        response = self.__CLIENT.upload_file(source_path, bucket_name, target_path,
        #ResponseCacheControl='no-cache'
        )
        """response = with open(source_path, 'rb') as data:
        return self.__CLIENT.upload_fileobj(data, bucket_name, target_path)"""
        logging.info('{0} put succeeded'.format(target_path))
        
    def _put_object(self,source:bytes,target_path:str,bucket_name:str=None):
        bucket_name=self._bucket_name_check(bucket_name)
        
        response = self.__CLIENT.put_object(Bucket=bucket_name,
            Key=target_path,
            Body=source)
        
    def put_object(self,source:ib.AbcMajorObject,format:str='json',bucket_name:str=None,check_file_exist:bool=True)->None:
        """insert/update target object

        args
        ---------
        source: AbcMajorObject, input object must be a instance of AbcMajorObject extends

        kwargs
        ---------

        bucket_name: str, aws bucket_name, if None, access the bucket in set_client() set set_client

        check_file_exist: bool , if True, raise NameError if exist; if False, overwrite object directly

        """
        if source.key==None: raise ValueError("AbcMajorObject.key==None")
        target_path=source.key
        
        target_key=self._gen_target_key(target_path,format=format)
        if check_file_exist:
            try:
                self._get_object(target_key,bucket_name)
            except NameError as e:
                # file not exist
                pass
            else:
                raise NameError('{0} exist'.format(target_key))
        
        sourceB=self.encrypt(source.dumpBytes(format=format))
        return  self._put_object(sourceB,target_key,bucket_name)

    def get_remote_object(self,target_path:str,format:str='json',bucket_name:str=None)->bytes:
        """get object in bucket

        args
        ---------
        target_path: str, target object path

        kwargs
        ---------

        bucket_name:str, aws bucket_name, if None, access the bucket in set_client() set set_client
        """
        target_key=self._gen_target_key(target_path,format=format)
        try:
            result=self._get_object(target_key,bucket_name)
        except NameError as e:
            logging.error(e)
            raise
        body=result['Body']._raw_stream.read()
        return self.decrypt(body)

    def _gen_target_key(self,target_path:str,format:str='json')->str:
        """generate encryption format to store target object

        args
        ---------
        target_path: str, target object path
        """
        extension=None
        if format=='json':
            extension='.json'
        elif format=='msgpack':
            extension='.msgpack'
        else:
            raise ValueError("no format {0}".format(format))

        return "{0}/{1}{2}".format(target_path,self.__ENCRYPTION,extension).replace("//","/")
    def encrypt(self,content:bytes)->bytes:
        """encrypt input, encrypt/decrypt factory is ENCRYPTION_MAP
        """
     
        if ENCRYPTION_MAP[self.__ENCRYPTION]["format"]==None:
            return content
        else:
            pwd=ENCRYPTION_MAP[self.__ENCRYPTION]["pwd"]
            fmt=ENCRYPTION_MAP[self.__ENCRYPTION]["format"]
            ec=ENCRYPTION_MAP[self.__ENCRYPTION]["encrypt"]
            return ec(content,pwd)
    def decrypt(self,content:bytes)->bytes:
        """decrypt input, encrypt/decrypt factory is ENCRYPTION_MAP
        """
        if ENCRYPTION_MAP[self.__ENCRYPTION]["format"]==None:
            return content
        else:
            pwd=ENCRYPTION_MAP[self.__ENCRYPTION]["pwd"]
            fmt=ENCRYPTION_MAP[self.__ENCRYPTION]["format"]
            de=ENCRYPTION_MAP[self.__ENCRYPTION]["decrypt"]
            
            return de(content,pwd)


    def download(self,target_list:typing.List[str],download_path:str,encrypt=False):
        def get_remote_object(key,extension):
            key=key.split('/key_')[0]
            return self.get_remote_object(key,format=extension)
        
        for target_key in target_list:
            local_dir=os.path.join(download_path,target_key).split('/key_')[0]
            local_file=os.path.join(download_path,target_key)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            extension=target_key.split('.')[-1]
            get=get_remote_object(target_key,extension)
            if encrypt:
                open(local_file,"wb").write(    self.encrypt(get  )   )
            else:
                if extension=='json':
                    with open(local_file,"w", encoding='utf-8') as f:
                        return f.write(  get.decode('utf-8')  )
                elif extension=='msgpack':
                    with open(local_file,"wb") as f:
                        return f.write(  get )
    def get_dict_from_msgpack_file(self,local_file_path):
        with open(local_file_path,"rb") as f:
            return msgpack.unpackb( self.decrypt(f.read()), encoding='utf-8')
    def get_dict_from_json_file(self,local_file_path):
        with open(local_file_path,"r", encoding='utf-8') as f:
            return json.loads( self.decrypt(f.read()), encoding='utf-8')
    def get_dict(self,local_file_path):
        extension=local_file_path.split('.')[-1]
        if extension=='json':
            return self.get_dict_from_json_file(local_file_path)
        elif extension=='msgpack':
            return self.get_dict_from_msgpack_file(local_file_path)