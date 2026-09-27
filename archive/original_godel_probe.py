#!/usr/bin/env python3
import argparse, json, math, random, hashlib
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def primes(n):
    out=[]; x=2
    while len(out)<n:
        ok=True
        r=int(math.sqrt(x))
        for p in out:
            if p>r: break
            if x%p==0:
                ok=False; break
        if ok: out.append(x)
        x += 1 if x==2 else 2
    return out


def make_pairs(n=80, seed=0):
    rng=random.Random(seed)
    letters=list("ABCDEFGHJKLMNPQRSTUVWXYZ")
    rows=[]
    for _ in range(n):
        a,b,c=rng.sample(letters,3)
        facts=f"{a} -> {b}\n{b} -> {c}"
        base=(
            "Follow the arrows until the chain ends. Return only the final capital letter.\n"
            + facts + "\n"
        )
        rows.append({"kind":"two", "prompt":base+f"Start: {a}\nFinal:", "answer":c, "triple":[a,b,c]})
        rows.append({"kind":"one", "prompt":base+f"Start: {b}\nFinal:", "answer":c, "triple":[a,b,c]})
    return rows


def sequence_logprob(model, tok, prompt, answer, device):
    # Score the full answer string; no generation randomness.
    p=tok(prompt, return_tensors="pt", add_special_tokens=False)
    a=tok(" "+answer, return_tensors="pt", add_special_tokens=False)
    input_ids=torch.cat([p.input_ids, a.input_ids], dim=1).to(device)
    with torch.no_grad():
        logits=model(input_ids=input_ids).logits
    plen=p.input_ids.shape[1]
    score=0.0
    for j in range(a.input_ids.shape[1]):
        pos=plen+j-1
        target=a.input_ids[0,j].to(device)
        score += torch.log_softmax(logits[0,pos], dim=-1)[target].item()
    return score


def answer_correct(model, tok, row, device, candidates):
    scores={c:sequence_logprob(model,tok,row["prompt"],c,device) for c in candidates}
    pred=max(scores,key=scores.get)
    return pred==row["answer"], pred, scores[row["answer"]]


def load_model(name, device):
    tok=AutoTokenizer.from_pretrained(name)
    kwargs={"torch_dtype":"auto"}
    try:
        model=AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", **kwargs)
    except TypeError:
        model=AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(device).eval()
    return tok,model


def get_attentions(model, tok, prompt, device):
    x=tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        try:
            out=model(**x, output_attentions=True, use_cache=False)
        except Exception:
            # Some recent HF models require eager attention at load time; if this fails,
            # the caller will report it rather than silently fabricating data.
            raise
    if out.attentions is None:
        raise RuntimeError("Model did not return attentions")
    return [a[0].detach().float().cpu() for a in out.attentions], x.input_ids[0].cpu()


def structural_factors(attentions, topk=1):
    """A-priori factors. No task label or token identity is used.

    E factor: local active edge (layer, head, query-position, key-position).
    P factor: 2-layer composition: q --(L,h)-> k --(L-1,h2)-> k2.
    We never multiply the giant Goedel number; the set/multiset of factors is its factorization.
    """
    factors=[]
    top_sources={}
    for L,a in enumerate(attentions):  # [H,T,T]
        H,T,_=a.shape
        for h in range(H):
            for q in range(1,T):
                legal=a[h,q,:q+1]
                k=min(topk, legal.numel())
                vals, idx=torch.topk(legal,k=k)
                srcs=tuple(int(i) for i in idx.tolist())
                top_sources[(L,h,q)]=srcs
                for s in srcs:
                    factors.append(("E",L,h,q,s))
    # recursive 2-layer ancestry factors
    for L in range(1,len(attentions)):
        H=attentions[L].shape[0]
        Hprev=attentions[L-1].shape[0]
        T=attentions[L].shape[1]
        for h in range(H):
            for q in range(1,T):
                for s in top_sources[(L,h,q)]:
                    if s==0: continue
                    for hp in range(Hprev):
                        for s2 in top_sources.get((L-1,hp,s),()):
                            factors.append(("P",L,h,q,s,hp,s2))
    return Counter(factors)


def assign_primes(all_factors):
    # Deterministic within a model/run: lexicographic factor order -> nth prime.
    fs=sorted(all_factors, key=repr)
    ps=primes(len(fs)+5)
    return {f:ps[i] for i,f in enumerate(fs)}


def exact_gcd_factorization(counters):
    if not counters: return Counter()
    common=set(counters[0])
    for c in counters[1:]: common &= set(c)
    out=Counter()
    for f in common: out[f]=min(c[f] for c in counters)
    return out


def prevalence(counters):
    n=len(counters); c=Counter()
    for x in counters: c.update(x.keys())
    return {f:c[f]/n for f in c}


def run_model(name, args):
    device=torch.device(args.device)
    print(f"\n=== {name} ===")
    tok,model=load_model(name,device)
    rows=make_pairs(args.pairs,args.seed)
    candidates=list("ABCDEFGHJKLMNPQRSTUVWXYZ")

    # Pair-level fairness: keep a chain only if the frozen pretrained model gets BOTH
    # the one-hop and two-hop query correct. No training/fine-tuning.
    by_triple={}
    for i,row in enumerate(rows): by_triple.setdefault(tuple(row['triple']),[]).append((i,row))
    kept=[]
    correct_map={}
    for triple,pair in by_triple.items():
        oks=[]
        for i,row in pair:
            ok,pred,_=answer_correct(model,tok,row,device,candidates)
            correct_map[i]=(ok,pred); oks.append(ok)
        if all(oks): kept.extend(i for i,_ in pair)

    print(f"paired-correct chains: {len(kept)//2}/{args.pairs}")
    if len(kept)//2 < args.min_correct_pairs:
        return {"model":name,"status":"too_weak","paired_correct":len(kept)//2}

    selected=[(i,rows[i]) for i in kept]
    random.Random(args.seed+1).shuffle(selected)
    # Split by chain, not individual query, to prevent leakage.
    triples=[]
    seen=set()
    for _,r in selected:
        t=tuple(r['triple'])
        if t not in seen: seen.add(t); triples.append(t)
    cut=max(1,len(triples)//2)
    discover=set(triples[:cut]); test=set(triples[cut:])

    rec=[]
    for j,(i,row) in enumerate(selected,1):
        att,ids=get_attentions(model,tok,row['prompt'],device)
        rec.append({"kind":row['kind'],"triple":tuple(row['triple']),"factors":structural_factors(att,args.topk)})
        if j%10==0: print(f"  extracted {j}/{len(selected)}")

    allf=set()
    for r in rec: allf.update(r['factors'])
    pmap=assign_primes(allf)

    def subset(split,kind):
        pool=discover if split=='discover' else test
        return [r['factors'] for r in rec if r['triple'] in pool and r['kind']==kind]

    d1,d2=subset('discover','one'),subset('discover','two')
    t1,t2=subset('test','one'),subset('test','two')
    g1,g2=exact_gcd_factorization(d1),exact_gcd_factorization(d2)
    unique1=set(g1)-set(g2); unique2=set(g2)-set(g1)

    # Exact Goedel-gcd rule: a test computation is tagged if it contains a factor
    # that was in the exact class-specific gcd on discovery data.
    def rate(counters, sig):
        if not counters: return None
        return sum(any(f in c for f in sig) for c in counters)/len(counters)

    exact={
        "two_on_two":rate(t2,unique2),
        "two_on_one":rate(t1,unique2),
        "one_on_one":rate(t1,unique1),
        "one_on_two":rate(t2,unique1),
        "gcd_one_factors":len(g1),
        "gcd_two_factors":len(g2),
        "unique_one_factors":len(unique1),
        "unique_two_factors":len(unique2),
    }

    # Exploratory near-gcd: factors present in >= threshold of discovery class and
    # <= (1-threshold) of the other class. This is NOT ML; it is a fixed frequency rule.
    pv1,pv2=prevalence(d1),prevalence(d2)
    th=args.near_gcd
    near2={f for f,v in pv2.items() if v>=th and pv1.get(f,0)<=1-th}
    near1={f for f,v in pv1.items() if v>=th and pv2.get(f,0)<=1-th}
    near={
        "threshold":th,
        "two_on_two":rate(t2,near2),"two_on_one":rate(t1,near2),
        "one_on_one":rate(t1,near1),"one_on_two":rate(t2,near1),
        "near_one_factors":len(near1),"near_two_factors":len(near2),
    }

    # Human-readable examples of prime assignments (we still never compute the monster integer).
    sample_two=sorted(list(near2 or unique2), key=repr)[:8]
    prime_examples=[{"prime":pmap[f],"factor":repr(f)} for f in sample_two]

    print("exact gcd test:",exact)
    print("near-gcd test:",near)
    return {
        "model":name,"status":"ok","paired_correct":len(kept)//2,
        "exact":exact,"near":near,"prime_examples":prime_examples,
        "note":"E=(local attention edge); P=(two-layer composed ancestry path). No task labels/token identities enter the factor definition."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+',default=['EleutherAI/pythia-70m','EleutherAI/pythia-160m','gpt2'])
    ap.add_argument('--pairs',type=int,default=40)
    ap.add_argument('--min-correct-pairs',type=int,default=8)
    ap.add_argument('--topk',type=int,default=1)
    ap.add_argument('--near-gcd',type=float,default=0.80)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--out',default='godel_probe_results.json')
    args=ap.parse_args()

    results=[]
    for name in args.models:
        try:
            results.append(run_model(name,args))
        except Exception as e:
            results.append({"model":name,"status":"error","error":repr(e)})
            print("ERROR",name,repr(e))
    with open(args.out,'w',encoding='utf-8') as f: json.dump(results,f,ensure_ascii=False,indent=2)
    print(f"\nSaved: {args.out}")

if __name__=='__main__':
    main()
