"""Atomic workflow state and immutable-output receipts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,default=str)
            stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def write_receipt(directory,name,metadata=None,files=None):
    directory=Path(directory)
    files=files if files is not None else [p for p in directory.rglob('*') if p.is_file() and p.name!=name]
    hashes={str(Path(p).relative_to(directory)):sha256(p) for p in files}
    atomic_json(directory/name,{'metadata':metadata or {},'sha256':hashes})


def verify_receipt(directory,name):
    directory=Path(directory)
    try:
        receipt=json.loads((directory/name).read_text())
        return bool(receipt['sha256']) and all((directory/p).is_file() and sha256(directory/p)==value
                                             for p,value in receipt['sha256'].items())
    except (ValueError,KeyError,FileNotFoundError):return False
