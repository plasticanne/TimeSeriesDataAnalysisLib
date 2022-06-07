JSON_R_data_v1_1='''{
    "info":{
        "ver":"0.0.1",
        "in_ver":"0.0.1",
        "fw_ver":"0.0.1",
        "sn":"A001",
        "patient":{
            "key": "/record/myprovider/1572940791.55",
            "index_id": "tw_zh_v1",
            "pt_id":"record01"
        }
    },
    "data":[
        {
            "measure_time":"2019-06-04T10:36:44Z",
            "put_time":"2019-06-04T10:36:44Z",
            "tag":null,
            "lat":24.95,
            "lon":121.16,
            "features":[
                {
                    "name": "rh",
                    "value": 90.11
                },
                {
                    "name": "tm",
                    "value": 11
                },
                {
                    "name": "p25",
                    "value": 50,
                    "cfg": [
                    9,
                    0.7,
                    40
                    ]
                }
            ]
        },
        {
            "measure_time":"2019-06-05T10:36:44",
            "put_time":"2019-06-05T10:36:44.016Z",
            "tag":null,
            "lat":24.95,
            "lon":121.16,
            "features":[
                {
                    "name": "rh",
                    "value": 90.11
                },
                {
                    "name": "tm",
                    "value": 11
                },
                {
                    "name": "p25",
                    "value": 50,
                    "cfg": [
                    2,
                    0.7,
                    40
                    ]
                }
            ]
        }
    ]
}'''
DICT_R_data_v1_2={
    "info":{
        "in_ver":"project1_v1_2",
        "fw_ver":"0.0.1",
        "sn":"A001",
        "link_set":"myprovider",
        "link_id":"record01",
    },
    "data":{
        "measure_time":["2019-06-04T10:36:44Z","2019-06-05T10:36:44"],
        "put_time":["2019-06-04T10:36:44Z","2019-06-05T10:36:44.016Z"],
        "tag":None,
        "lat":[24.95,24.95],
        "lon":[121.16,121.16],
        "cfg": [[None,None,[9,0.7,40]],[None,None,[2,0.7,40]]],
        "shape":[2,3],
        "value":[ [90.11,11,50],[90.11,11,50]],
    },
    "annotation":{
        "label":None,
        "label_set":None,
        "feature_name":["rh","tm","p25"],
        "mark":None,
    }
    

}
