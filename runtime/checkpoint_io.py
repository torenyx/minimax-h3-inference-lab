from contextlib import ExitStack
import math,json,struct
from pathlib import Path
import torch
from safetensors import safe_open

DTYPES = {'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E5M2':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}

def header(path):
    with path.open('rb') as handle:
        size=struct.unpack('<Q',handle.read(8))[0]
        catalog=json.loads(handle.read(size))
    return size+8,{k:v for k,v in catalog.items() if k!='__metadata__'}


class Reader:
    def __init__(self,paths):
        self.stack=ExitStack()
        self.catalog={}
        self.locations={}
        self.files={}
        for path in paths:
            start,entries=header(path)
            tensors=self.stack.enter_context(safe_open(str(path),framework='pt',device='cpu'))
            handle=self.stack.enter_context(path.open('rb'))
            for name,entry in entries.items():
                if name in self.catalog:
                    raise ValueError('Duplicate tensor '+name)
                self.catalog[name]=entry
                self.locations[name]=(handle,start)
                self.files[name]=tensors

    def tensor(self,name):
        return self.files[name].get_tensor(name)

    def copy(self,name,target):
        handle,start=self.locations[name]
        begin,end=self.catalog[name]['data_offsets']
        handle.seek(start+begin)
        remaining=end-begin
        while remaining:
            data=handle.read(min(remaining,16*1024*1024))
            if not data:
                raise EOFError(name)
            target.write(data)
            remaining-=len(data)

    def close(self):
        self.stack.close()


def write_tensor(stream,tensor):
    raw=tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    data=memoryview(raw).cast('B')
    assert stream.write(data)==len(data)


def make_catalog(specs):
    offset=0
    catalog={}
    for name,dtype,shape in specs:
        size=math.prod(shape)*DTYPES[dtype]
        catalog[name]={'dtype':dtype,'shape':list(shape),'data_offsets':[offset,offset+size]}
        offset+=size
    return catalog,offset


def start_output(path,catalog,metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    raw=json.dumps({'__metadata__':{'format':'pt',**metadata},**catalog},separators=(',',':')).encode()
    raw+=b' '*((-len(raw))%8)
    handle=temporary.open('wb')
    handle.write(struct.pack('<Q',len(raw)))
    handle.write(raw)
    return handle,temporary,8+len(raw)

