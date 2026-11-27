"""Fine-tune pretrained wav2vec2-base with SentencePiece CTC targets."""
from __future__ import annotations
import argparse, json, math, random, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .metrics import corpus_error_rates, greedy_decode
from .tokenizer import ASRTokenizer
from .wav2vec2_model import Wav2Vec2CTC
from .data import DurationBucketBatchSampler

ROOT = Path(__file__).resolve().parents[2]
def resolve(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p

class WaveformDataset(Dataset):
    def __init__(self, manifest, tokenizer, sample_rate, max_samples=None):
        self.tokenizer,self.sample_rate,self.items=tokenizer,int(sample_rate),[]
        with resolve(manifest).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item=json.loads(line)
                if not item.get("text","").strip(): continue
                ap=Path(item["audio_filepath"])
                if not ap.is_file(): raise FileNotFoundError(f"Manifest audio missing: {ap}")
                item["duration"]=float(item.get("duration",sf.info(str(ap)).duration))
                self.items.append(item)
                if max_samples and len(self.items)>=int(max_samples): break
        if not self.items: raise ValueError(f"No usable audio in manifest {manifest}")
        self.durations=[max(.1,x["duration"]) for x in self.items]
    def __len__(self): return len(self.items)
    def __getitem__(self,index):
        item=self.items[index]
        audio,sr=sf.read(item["audio_filepath"],dtype="float32",always_2d=True)
        wave=audio.mean(axis=1)
        if sr!=self.sample_rate: wave=resample_poly(wave,self.sample_rate,sr).astype(np.float32)
        wave=wave.astype(np.float32)
        wave=(wave-wave.mean())/max(float(wave.std()),1e-7)
        ids=self.tokenizer.encode(item["text"])
        if not ids: raise ValueError(f"Empty tokens: {item.get('id',index)}")
        return {"id":str(item.get("id",index)),"waveform":torch.from_numpy(wave.copy()),
                "tokens":torch.tensor(ids,dtype=torch.long),"text":item["text"],
                "duration":len(wave)/self.sample_rate}

def collate_waveforms(batch):
    lengths=torch.tensor([x["waveform"].numel() for x in batch],dtype=torch.long)
    waves=torch.zeros(len(batch),int(lengths.max()),dtype=torch.float32)
    for i,x in enumerate(batch): waves[i,:lengths[i]]=x["waveform"]
    tl=torch.tensor([x["tokens"].numel() for x in batch],dtype=torch.long)
    return {"ids":[x["id"] for x in batch],"texts":[x["text"] for x in batch],
            "durations":[x["duration"] for x in batch],"waveforms":waves,
            "waveform_lengths":lengths,"targets":torch.cat([x["tokens"] for x in batch]),
            "target_lengths":tl}

def evaluate(model,loader,tokenizer,device,criterion,amp):
    model.eval(); refs=[]; hyps=[]; loss_sum=0.; count=0; impossible=0
    with torch.inference_mode():
        for b in tqdm(loader,desc="Validation",leave=False):
            waves=b["waveforms"].to(device); lengths=b["waveform_lengths"].to(device)
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
                logits,outlen=model(waves,lengths)
            targets=b["targets"].to(device); tl=b["target_lengths"]
            impossible+=int((outlen.cpu()<tl).sum())
            lp=F.log_softmax(logits.float(),dim=-1)
            loss=criterion(lp.transpose(0,1),targets,outlen.cpu(),tl)
            loss_sum+=float(loss)*len(b["texts"]); count+=len(b["texts"])
            hyps.extend(greedy_decode(logits.float(),outlen,tokenizer)); refs.extend(b["texts"])
    rates=corpus_error_rates(refs,hyps)
    return {"loss":loss_sum/max(count,1),"wer":rates["wer"],"cer":rates["cer"],
            "references":refs,"hypotheses":hyps,"ctc_impossible":impossible}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=Path("training/asr/config.yaml"))
    p.add_argument("--epochs",type=int,default=None)
    p.add_argument("--max-train-samples",type=int,default=None)
    p.add_argument("--max-valid-samples",type=int,default=None)
    args=p.parse_args()
    with resolve(args.config).open(encoding="utf-8") as f: cfg=yaml.safe_load(f)
    tr=cfg["training"]
    if args.epochs: tr["epochs"]=args.epochs
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed=int(cfg.get("seed",42)); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=True; torch.cuda.reset_peak_memory_stats()
    outdir=resolve(cfg["output"]["directory"]); outdir.mkdir(parents=True,exist_ok=True)
    tok=ASRTokenizer(resolve(cfg["data"]["tokenizer_model"]))
    train=WaveformDataset(cfg["data"]["train_manifest"],tok,cfg["audio"]["sample_rate"],
        args.max_train_samples or cfg["data"].get("max_train_samples"))
    valid=WaveformDataset(cfg["data"]["valid_manifest"],tok,cfg["audio"]["sample_rate"],
        args.max_valid_samples or cfg["data"].get("max_valid_samples"))
    sampler=DurationBucketBatchSampler(train.durations,float(tr.get("max_batch_seconds",20)),
        int(tr.get("max_batch_size",8)),shuffle=True,seed=seed)
    workers=int(cfg["data"].get("num_workers",2))
    train_loader=DataLoader(train,batch_sampler=sampler,num_workers=workers,
        pin_memory=device.type=="cuda",persistent_workers=workers>0,collate_fn=collate_waveforms)
    valid_loader=DataLoader(valid,batch_size=int(tr.get("valid_batch_size",4)),shuffle=False,
        num_workers=workers,pin_memory=device.type=="cuda",collate_fn=collate_waveforms)
    mc={"backend":"wav2vec2","vocab_size":tok.vocab_size,
        "freeze_encoder_layers":int(cfg["model"].get("freeze_encoder_layers",8)),
        "dropout":float(cfg["model"].get("dropout",.1))}
    model=Wav2Vec2CTC(**mc,pretrained=True).to(device)
    print("Loaded torchaudio WAV2VEC2_BASE pretrained weights.")
    print(f"Trainable parameters: {sum(x.numel() for x in model.parameters() if x.requires_grad):,}")
    print(f"Frozen Transformer layers: {mc['freeze_encoder_layers']} / 12")
    groups=model.parameter_groups(float(tr.get("encoder_learning_rate",1e-5)),
        float(tr.get("head_learning_rate",5e-4)),float(tr.get("weight_decay",.01)))
    opt=torch.optim.AdamW(groups,betas=(.9,.98),eps=1e-8)
    accum=max(1,int(tr.get("grad_accum_steps",4)))
    total=max(1,math.ceil(len(train_loader)/accum)*int(tr["epochs"]))
    warm=max(1,int(tr.get("warmup_steps",300)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/warm)*
        (.5+.5*math.cos(math.pi*min(s,total)/total)))
    amp=device.type=="cuda" and bool(tr.get("amp",True))
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    ctc=nn.CTCLoss(blank=tok.blank_id,zero_infinity=True)
    best=float("inf"); stale=0; history=[]
    for epoch in range(1,int(tr["epochs"])+1):
        sampler.set_epoch(epoch); model.train(); opt.zero_grad(set_to_none=True)
        running=0.; seen=0; updates=0
        bar=tqdm(train_loader,desc=f"Epoch {epoch}")
        for bi,b in enumerate(bar):
            waves=b["waveforms"].to(device,non_blocking=True)
            lens=b["waveform_lengths"].to(device,non_blocking=True)
            targets=b["targets"].to(device,non_blocking=True); tl=b["target_lengths"]
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp):
                logits,outlen=model(waves,lens)
            # Never silently count CTC-impossible examples as zero loss.
            keep=outlen.cpu()>=tl
            if not keep.all():
                parts=[]; off=0
                for i,n in enumerate(tl.tolist()):
                    if keep[i]: parts.append(targets[off:off+n])
                    off+=n
                if not keep.any(): continue
                targets=torch.cat(parts); tl=tl[keep]; outlen=outlen[keep.to(device)]
                logits=logits[keep.to(device)]
            lp=F.log_softmax(logits.float(),dim=-1)
            loss=ctc(lp.transpose(0,1),targets,outlen.cpu(),tl)
            if not torch.isfinite(loss): continue
            scaler.scale(loss/accum).backward()
            running+=float(loss.detach())*int(keep.sum()); seen+=int(keep.sum())
            if (bi+1)%accum==0 or bi+1==len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(tr.get("max_grad_norm",1.)))
                before=scaler.get_scale(); scaler.step(opt); scaler.update()
                stepped=(not scaler.is_enabled()) or scaler.get_scale()>=before
                opt.zero_grad(set_to_none=True)
                if stepped: sched.step(); updates+=1
            bar.set_postfix(loss=f"{loss.item():.3f}",head_lr=f"{opt.param_groups[1]['lr']:.1e}")
        val=evaluate(model,valid_loader,tok,device,ctc,amp)
        row={"epoch":epoch,"train_loss":running/max(seen,1),"valid_loss":val["loss"],
             "wer":val["wer"],"cer":val["cer"],"ctc_impossible":val["ctc_impossible"],
             "optimizer_updates":updates}
        history.append(row); print(json.dumps(row))
        payload={"model_architecture_version":3,"epoch":epoch,"model_state_dict":model.state_dict(),
            "model_config":mc,"audio_config":cfg["audio"],"tokenizer_vocab_size":tok.vocab_size,
            "backend":"wav2vec2","validation":row}
        torch.save(payload,outdir/"last.pt")
        if val["wer"]<best:
            best=val["wer"]; stale=0; torch.save(payload,outdir/"best.pt")
        else: stale+=1
        (outdir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        if device.type=="cuda":
            print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
            torch.cuda.reset_peak_memory_stats()
        if stale>=int(tr.get("early_stopping_patience",6)):
            print("Early stopping: validation WER did not improve."); break
    shutil.copy2(resolve(cfg["data"]["tokenizer_model"]),outdir/"tokenizer.model")
    print(f"Best checkpoint: {outdir/'best.pt'}")

if __name__=="__main__": main()
