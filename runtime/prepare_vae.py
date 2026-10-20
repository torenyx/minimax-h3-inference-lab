from pathlib import Path
import re,json,torch
from runtime.checkpoint_io import Reader,make_catalog,start_output,write_tensor
import os
r=Path(os.environ['H3_CACHE_DIR'])
b=Path(os.environ['H3_ORIGINAL_MODEL'])
d=r/'video-vae';d.mkdir(parents=True,exist_ok=True)
wrapper=json.loads((b/'video_vae/config.json').read_text());original=json.loads((b/'video_vae/source/config.json').read_text())
config={**wrapper,'in_channels':original['in_channels'],'out_channels':original['out_ch'],'block_out_channels':[original['ch']*x for x in original['ch_mult']],'layers_per_block':original['num_res_blocks'],'spatial_downsample_factors':original['space_down'],'temporal_downsample_factors':original['time_down']}
(d/'config.json').write_text(json.dumps(config,indent=2))
reader=Reader([b/'video_vae/source/model.safetensors']);specs=[];ops=[]
for name,entry in reader.catalog.items():
 if name=='decoder.mask_token':continue
 target=name.replace('decoder.x_embedder.','decoder.proj_in.')
 target=re.sub(r'encoder.down.(\d+).block.(\d+).',r'encoder.down_blocks.\1.resnets.\2.',target)
 target=re.sub(r'encoder.down.(\d+).downsample.',r'encoder.down_blocks.\1.downsamplers.0.',target)
 target=target.replace('.nin_shortcut.','.conv_shortcut.')
 if '.attn.to_qkv.' in name:
  shape=entry['shape'];assert shape[0]==32*3*64
  for part,letter in enumerate('qkv'):
   specs.append((target.replace('.to_qkv.','.to_'+letter+'.'),entry['dtype'],[shape[0]//3,*shape[1:]]));ops.append((name,'qkv',part))
 else:
  target=target.replace('.attn.to_out.','.attn.to_out.0.').replace('.ff.w1.','.ff.net.0.proj.').replace('.ff.w2.','.ff.net.2.')
  specs.append((target,entry['dtype'],entry['shape']));ops.append((name,'swap' if '.ff.w1.' in name else 'copy',None))
p=d/'model.safetensors'
if not p.exists():
 cat,size=make_catalog(specs);stream,temp,start=start_output(p,cat,{'source':'original H3 VAE; layout-only conversion'})
 with stream:
  for i,(name,op,part) in enumerate(ops):
   if op=='copy':reader.copy(name,stream)
   else:
    t=reader.tensor(name)
    if op=='swap':a,z=t.chunk(2,0);t=torch.cat([z,a],0)
    else:t=t.view(32,3,64,*t.shape[1:])[:,part].flatten(0,1)
    write_tensor(stream,t.contiguous())
   if i%150==0:print('VAE_CONVERT',i,len(ops),flush=True)
  assert stream.tell()==start+size
 temp.replace(p)
reader.close()
print('NATIVE_VIDEO_VAE_READY',flush=True)
