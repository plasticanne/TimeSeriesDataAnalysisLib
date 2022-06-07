
from Crypto.Cipher import AES
import base64
from binascii import b2a_hex, a2b_hex
from Crypto import Random

_IV = 8 * '\x02'.encode('utf-8')
BLOCK_SIZE = 8
PAK=b2a_hex   #base64.b64encode
UNPAK=a2b_hex #base64.b64decode
def pad(data:bytes):
    length = BLOCK_SIZE - (len(data) % BLOCK_SIZE)
    return data + (b'\0' * length)
def pkcs7_padding(data):
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import algorithms
    if not isinstance(data, bytes):
        data = data.encode()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    return padded_data
def aes_encrypt(data:bytes, pwd:str)-> bytes:
    cryptor = AES.new(pwd.encode('utf-8'), AES.MODE_CBC, _IV)
    return PAK(cryptor.encrypt(pad(data)))

def aes_decrypt(data:bytes, pwd:str) -> str:
    data=UNPAK( data)
    cryptor = AES.new(pwd.encode('utf-8'), AES.MODE_CBC, _IV)
    return cryptor.decrypt( data ).rstrip(b'\0')

def aes_encrypt_dy(data:bytes, pwd:str):
    IV = Random.new().read(BLOCK_SIZE)
    cryptor = AES.new(pwd.encode('utf-8'), AES.MODE_CBC, IV)
    return PAK(IV + cryptor.encrypt(pad(data)))

def aes_decrypt_dy(data:bytes, pwd:str):
    data= UNPAK(data)
    IV = data[:BLOCK_SIZE]
    cryptor = AES.new(pwd.encode('utf-8'), AES.MODE_CBC, IV)
    return cryptor.decrypt(data[BLOCK_SIZE:]).rstrip(b'\0')

ENCRYPTION_MAP={
    "key_000":{
        "format":None
    },
    "key_001":{
        "format":"AES-256-CBC+b2a_hex",
        "pwd":"123454654898",
        "encrypt":aes_encrypt,
        "decrypt":aes_decrypt
    }


    
}


