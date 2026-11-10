import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0,str(Path.cwd()))
from gvc_llm.model import GenerativeCritic, posterior

torch.set_num_threads(4)
torch.manual_seed(1)
out=Path('agent_reports/beta-kl-checkpoint340')
c=torch.load('outputs/gvc/critic.pt',map_location='cpu',weights_only=False)
cfg=c['config']; print('checkpoint',c['step'],flush=True)
tok=AutoTokenizer.from_pretrained(cfg['model'])
policy=AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
critic=GenerativeCritic(AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda(),cfg['lora_rank']).eval()
critic.load_state_dict(c['trainable'],strict=False)
data=pd.read_parquet(cfg['train_file'])
samples=[json.loads(l) for l in open('outputs/gvc/samples.jsonl')]
samples=[s for s in samples if s['step']==c['step']]
logs=[]; qs=[]; weights=[]
with torch.inference_mode():
 for i,s in enumerate(samples):
  prompt=data.iloc[s['index']]['prompt'][0]['content']
  p=tok.encode(prompt); r=tok.encode(s['response'],add_special_tokens=False)
  if not r: continue
  ids=torch.tensor([p+r],device='cuda'); mask=torch.ones_like(ids)
  inds=torch.tensor(np.unique(np.linspace(len(p)-1,len(p)+len(r)-2,32).astype(int)),device='cuda')
  h=policy.model(input_ids=ids,attention_mask=mask).last_hidden_state[0,inds]
  lp=policy.lm_head(h).float().log_softmax(-1)
  ch,_=critic.features(ids.repeat(2,1),mask.repeat(2,1),torch.tensor([0,1],device='cuda'))
  q=posterior(critic.logits(ch[0,inds]),critic.logits(ch[1,inds]),critic.prior(ch[1,inds].float()))
  logs.append(lp); qs.append(q); weights.extend([len(r)/len(inds)]*len(inds))
  print('sequence',i,'length',len(r),flush=True)
 lp=torch.cat(logs); q=torch.cat(qs); w=torch.tensor(weights,device='cuda'); w=w/w.sum()
 def kl(beta):
  lt=(lp+q/beta).log_softmax(-1)
  return (lt.exp()*(lt-lp)).sum(-1)
 lo,hi=0.00001,1.
 for _ in range(28):
  mid=(lo+hi)/2
  if (kl(mid)*w).sum().item()>1: lo=mid
  else: hi=mid
 beta=(lo+hi)/2
 k=kl(beta)
 result={'checkpoint_step':c['step'],'run':'xcwfnqjr','sequences':len(samples),'prefixes':len(w),'weighting':'response-token weighted; 32 evenly spaced prefixes per sequence','direction':'KL(tilted || base), nats per token','beta':beta,'mean_kl':(k*w).sum().item(),'prefix_kl_quantiles':torch.quantile(k,torch.tensor([0.,.1,.5,.9,1.],device='cuda')).tolist(),'sweep':{str(b):(kl(b)*w).sum().item() for b in [1.,.1,.05,.02,.01,.005,beta]}}
 # Independent halves show sensitivity to sampled trajectories.
 for label,sl in [('first_half',slice(0,len(w)//2)),('second_half',slice(len(w)//2,None))]:
  lo,hi=.00001,1.
  for _ in range(24):
   mid=(lo+hi)/2
   val=(kl(mid)[sl]*w[sl]).sum()/w[sl].sum()
   if val.item()>1: lo=mid
   else: hi=mid
  result[label+'_beta']=(lo+hi)/2
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
