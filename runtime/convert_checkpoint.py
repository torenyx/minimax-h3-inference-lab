"""Derive Comfy-compatible rank-8 BF16 and INT8+ConvRot from original shards.

The rank-8 construction follows Wan2GP prune_checkpoint.py. Unchanged tensors
are streamed without loading a whole checkpoint; original QKV rows are explicitly
reordered to the Comfy loader's layout. Originals are never opened for writing.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time

from runtime.compat import enable
enable()
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

DTYPES = {'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E5M2':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}
QUANT_WEIGHT = re.compile(r'blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$')


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


def loader_hash(catalog):
    # DiffSynth fingerprints names/shapes, not checkpoint contents.
    names=[]
    for name,entry in catalog.items():
        names.extend([name,name+':'+ '_'.join(map(str,entry['shape']))])
    return hashlib.md5(','.join(sorted(names)).encode()).hexdigest()


def curve(reader,t,device):
    weights=[reader.tensor('time_embedder.'+name).to(device=device,dtype=torch.float64) for name in ['proj_in.weight','proj_in.bias','proj_out.weight','proj_out.bias']]
    w1,b1,w2,b2=weights
    half=w1.shape[1]//2
    frequencies=torch.exp(-math.log(10000.)*torch.arange(half,device=device,dtype=torch.float32)/half)
    args=t.float()[:,None]*frequencies[None]
    embedding=torch.cat([args.cos(),args.sin()],-1).double()
    return F.silu(F.linear(F.silu(F.linear(embedding,w1,b1)),w2,b2))


@torch.inference_mode()
def prune(source,output,device,rank=8,grid=1025,row_chunk=2048):
    config=json.loads((source/'config.json').read_text())
    index=json.loads((source/'model.safetensors.index.json').read_text())
    reader=Reader([source/name for name in sorted(set(index['weight_map'].values()))])
    started=time.monotonic()
    try:
        t=torch.linspace(0,1,grid,device=device)
        values=curve(reader,t,device)
        mean=values.mean(0)
        u,s,vh=torch.linalg.svd(values-mean,full_matrices=False)
        table=(u[:,:rank]*s[:rank]).float()
        basis=vh[:rank].float()
        mean=mean.float()
        reconstructed=table.double()@basis.double()+mean.double()
        relative_curve=((reconstructed-values).square().sum()/values.square().sum()).sqrt().item()
        if relative_curve>1e-4:
            raise RuntimeError(f'Curve approximation error is too large: {relative_curve}')
        specs=[]
        for name,entry in sorted(reader.catalog.items()):
            if name.startswith('time_embedder.') or name.endswith('.adaln_proj.linear.weight'):
                continue
            if name.endswith('.adaln_proj.linear.bias'):
                specs.append((name,'F32',entry['shape']))
                weight=name[:-4]+'weight'
                specs.append((weight,'F32',[reader.catalog[weight]['shape'][0],rank]))
            else:
                specs.append((name,entry['dtype'],entry['shape']))
        specs.append(('adaln_t_table','F32',[grid,rank]))
        catalog,size=make_catalog(specs)
        stream,temp,data_start=start_output(output,catalog,{'h3_source':'original-shards','adaln_curve_rank':str(rank),'adaln_curve_grid':str(grid),'qkv_layout':'comfy-qkv-head'})
        probes=torch.linspace(.031,.969,17,device=device)
        exact=curve(reader,probes,device).float()
        pos=probes*(grid-1)
        lo=pos.floor().long().clamp(max=grid-2)
        coordinates=torch.lerp(table[lo],table[lo+1],(pos-lo).unsqueeze(1))
        squared_error=squared_reference=0.
        projections=0
        with stream:
            pending={}
            for name,_,_ in specs:
                if name=='adaln_t_table':
                    write_tensor(stream,table)
                elif name.endswith('.adaln_proj.linear.bias'):
                    weight_name=name[:-4]+'weight'
                    original=reader.tensor(weight_name)
                    bias=reader.tensor(name)
                    folded=torch.empty(original.shape[0],rank,dtype=torch.float32)
                    folded_bias=torch.empty(original.shape[0],dtype=torch.float32)
                    for start in range(0,original.shape[0],row_chunk):
                        stop=min(start+row_chunk,original.shape[0])
                        w=original[start:stop].to(device=device,dtype=torch.float32)
                        b=bias[start:stop].to(device=device,dtype=torch.float32)
                        small=w@basis.T
                        offset=b+w@mean
                        folded[start:stop].copy_(small.cpu())
                        folded_bias[start:stop].copy_(offset.cpu())
                        expected=F.linear(exact,w,b)
                        actual=F.linear(coordinates,small,offset)
                        squared_error+=(actual-expected).double().square().sum().item()
                        squared_reference+=expected.double().square().sum().item()
                    write_tensor(stream,folded_bias)
                    pending[weight_name]=folded
                    projections+=1
                    print(f'Pruned AdaLN {projections}/{config["num_layers"]+1}: {name}',flush=True)
                elif name.endswith('.adaln_proj.linear.weight'):
                    write_tensor(stream,pending.pop(name))
                elif name.endswith('attn.qkv_proj.weight'):
                    w=reader.tensor(name)
                    heads=config['num_attention_heads']
                    dim=config['attention_head_dim']
                    reordered=w.reshape(heads,3,dim,w.shape[1]).permute(1,0,2,3).reshape(w.shape).contiguous()
                    write_tensor(stream,reordered)
                else:
                    reader.copy(name,stream)
            assert stream.tell()==data_start+size
        assert projections==config['num_layers']+1
        modulation_error=math.sqrt(squared_error/squared_reference)
        if modulation_error>0.01:
            raise RuntimeError(f'Pruning verification failed: {modulation_error}')
        temp.replace(output)
        save_file({'basis':basis.cpu().contiguous(),'mean':mean.cpu().contiguous(),'table':table.cpu().contiguous()},str(output.with_suffix('.basis.safetensors')))
        return {'state':'passed','path':str(output),'bytes':output.stat().st_size,'loader_hash':loader_hash(catalog),'projection_count':projections,'curve_relative_rmse':relative_curve,'modulation_relative_rmse':modulation_error,'elapsed_seconds':time.monotonic()-started,'qkv_reordered':True}
    finally:
        reader.close()


@torch.inference_mode()
def recover_pruned_basis(source,output,device):
    """The complete weight file was renamed before the small basis save failed."""
    index=json.loads((source/'model.safetensors.index.json').read_text())
    reader=Reader([source/name for name in sorted(set(index['weight_map'].values()))])
    data_start,catalog=header(output)
    try:
        assert output.stat().st_size==data_start+max(e['data_offsets'][1] for e in catalog.values())
        projections=[name for name in catalog if name.endswith('.adaln_proj.linear.weight')]
        assert len(projections)==51 and all(catalog[name]['shape'][1]==8 for name in projections)
        with safe_open(str(output),framework='pt',device='cpu') as tensors:
            table=tensors.get_tensor('adaln_t_table').contiguous()
        values=curve(reader,torch.linspace(0,1,table.shape[0],device=device),device).cpu()
        design=torch.cat([table.double(),torch.ones(table.shape[0],1,dtype=torch.float64)],dim=1)
        affine=torch.linalg.lstsq(design,values,driver='gelsd').solution
        reconstructed=design@affine
        relative=((reconstructed-values).square().sum()/values.square().sum()).sqrt().item()
        assert relative<1e-4,relative
        save_file({'basis':affine[:-1].float().contiguous(),'mean':affine[-1].float().contiguous(),'table':table},str(output.with_suffix('.basis.safetensors')))
        return {'state':'passed','path':str(output),'bytes':output.stat().st_size,'loader_hash':loader_hash(catalog),'projection_count':51,'curve_relative_rmse':relative,'qkv_reordered':True,'basis_recovered_after_complete_weight_write':True,'modulation_verified_before_atomic_rename':True}
    finally:
        reader.close()


@torch.inference_mode()
def quantize(source,output,device,row_chunk=1024):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor
    ck.registry.disable('cuda')
    reader=Reader([source])
    started=time.monotonic()
    marker=json.dumps({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}).encode()
    try:
        specs=[]
        for name,entry in reader.catalog.items():
            if QUANT_WEIGHT.fullmatch(name):
                rows,columns=entry['shape']
                assert columns%256==0,name
                prefix=name[:-7]
                specs.extend([(name,'I8',[rows,columns]),(prefix+'.weight_scale','F32',[rows,1]),(prefix+'.comfy_quant','U8',[len(marker)])])
            else:
                specs.append((name,entry['dtype'],entry['shape']))
        catalog,size=make_catalog(specs)
        stream,temp,data_start=start_output(output,catalog,{'h3_source':source.name,'h3_quantization':'int8-convrot','h3_quant_backend':'comfy-kitchen-0.2.37'})
        quantized=0
        with stream:
            for name,entry in reader.catalog.items():
                if not QUANT_WEIGHT.fullmatch(name):
                    reader.copy(name,stream)
                    continue
                weight=reader.tensor(name)
                scales=torch.empty(weight.shape[0],1,dtype=torch.float32)
                for start in range(0,weight.shape[0],row_chunk):
                    stop=min(start+row_chunk,weight.shape[0])
                    row=weight[start:stop].to(device=device,dtype=torch.bfloat16)
                    qt=QuantizedTensor.from_float(row,'TensorWiseINT8Layout',is_weight=True,per_channel=True,convrot=True,convrot_groupsize=256)
                    write_tensor(stream,qt._qdata)
                    scales[start:stop].copy_(qt._params.scale.float().cpu())
                    del qt,row
                write_tensor(stream,scales)
                stream.write(marker)
                quantized+=1
                if quantized%10==0:
                    print(f'Quantized {quantized} INT8+ConvRot matrices',flush=True)
            assert stream.tell()==data_start+size
        assert quantized==200,quantized
        temp.replace(output)
        return {'state':'passed','path':str(output),'bytes':output.stat().st_size,'loader_hash':loader_hash(catalog),'quantized_matrices':quantized,'weight_dtype':'int8','activation_dtype':'dynamic int8','elapsed_seconds':time.monotonic()-started}
    finally:
        reader.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['FL2VA','Ref2VA'],required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32=False
    source=args.model_dir/'transformer'
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    report_path=output/'conversion.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'variant':args.variant}
    for mode in ['pruned_bf16','pruned_int8_convrot']:
        destination=output/(mode+'.safetensors')
        if destination.exists() and report.get(mode,{}).get('state')=='passed':
            continue
        try:
            if mode=='pruned_bf16':
                if destination.exists() and 'non contiguous tensor' in report.get(mode,{}).get('error',''):
                    report[mode]=recover_pruned_basis(source,destination,args.device)
                else:
                    report[mode]=prune(source,destination,args.device)
            else:
                report[mode]=quantize(output/'pruned_bf16.safetensors',destination,args.device)
            report_path.write_text(json.dumps(report,indent=2))
            print(json.dumps(report[mode]),flush=True)
        except Exception as error:
            report[mode]={'state':'failed','error':repr(error)}
            report_path.write_text(json.dumps(report,indent=2))
            raise


if __name__=='__main__':
    main()
