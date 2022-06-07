
#import calendar
from dateutil.relativedelta import relativedelta as relativedelta
from dateutil.parser import parse
from dateutil.tz import tzlocal
import time
from datetime import datetime,timezone

def timestamp_utc_now()->float:
    return round(time.time() * 1000)/1000

def any_datetime_2_utc_timestamp(timeN:datetime)->float :
    if timeN.tzinfo==None:
        strptime=time.mktime(timeN.timetuple())
        return round(strptime* 1000)/1000
    else:
        timeN=timeN.astimezone(timezone.utc)
        return round(timeN.timestamp()* 1000)/1000
    
def utc_timestamp_2_utc_datetime(timeN:float)->datetime:
    return datetime.utcfromtimestamp( timeN).replace(tzinfo=timezone.utc)
def utc_timestamp_2_loaclzone_datetime(timeN:float)->str:
    return datetime.fromtimestamp(timeN).astimezone(tzlocal())
def any_datetime_2_utc_isoformat(timeN:datetime)->str:
    return timeN.astimezone(timezone.utc).isoformat(timespec='milliseconds')#.replace('+00:00', 'Z')

def any_datetime_2_loaclzone_isoformat(timeN:datetime)->str:
    return timeN.astimezone(tzlocal()).isoformat(timespec='milliseconds')

def any_isoformat_2_utc_datetime(timeN:str)->datetime:
    return parse(timeN).astimezone(timezone.utc)
def utc_timestamp_2_utc_isoformat(timeN:float)->str:
    return any_datetime_2_utc_isoformat(utc_timestamp_2_utc_datetime(timeN))
def utc_timestamp_2_loaclzone_isoformat(timeN:float)->str:
    return any_datetime_2_loaclzone_isoformat(utc_timestamp_2_utc_datetime(timeN))

def any_isoformat_2_utc_timestamp(timeN:str)->float:
    return any_datetime_2_utc_timestamp(any_isoformat_2_utc_datetime(timeN))
def yyyy_mm(m_time:datetime)->str:
    return "{0}_{1:0>2d}".format(m_time.year,m_time.month)
def get_local_date_now():
    return datetime.now().astimezone(tzlocal())
def local_date_to_filename(timeN:datetime):
    return timeN.replace(tzinfo=None).isoformat(timespec="milliseconds").replace(":","-")
def utc_datetime_loaclzone_filename(timeN:datetime):
    return timeN.astimezone(tzlocal()).replace(tzinfo=None).isoformat(timespec="milliseconds").replace(":","-")
def loaclzone_filename_to_loaclzone_datetime(filename:str):
    return datetime.strptime(filename, '%Y-%m-%dT%H-%M-%S.%f')


    

