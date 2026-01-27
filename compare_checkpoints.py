import torch
import torch.distributed.checkpoint as dcp
import sys
import os

def load_dcp(path):
    state_dict = {}
    dcp.load(state_dict, checkpoint_id=path)
    return state_dict

def compare(path1, path2):
    print(f"Comparing {path1} and {path2}...")
    sd1 = load_dcp(path1)
    sd2 = load_dcp(path2)
    
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    
    if keys1 != keys2:
        print("FAIL: Key mismatch!")
        print("In 1 but not 2:", keys1 - keys2)
        print("In 2 but not 1:", keys2 - keys1)
        return False
        
    all_match = True
    for k in keys1:
        t1 = sd1[k]
        t2 = sd2[k]
        if not torch.equal(t1, t2):
            print(f"FAIL: Value mismatch for {k}")
            print(f"Max diff: {(t1 - t2).abs().max()}")
            all_match = False
            
    if all_match:
        print("SUCCESS: All weights match exactly!")
        return True
    else:
        return False

if __name__ == "__main__":
    p1 = "/gpfs/gibbs/pi/krishnaswamy_smita/cl2482/Transformer-Dispersion/transformer_dispersion/pretrain_qwen3_huggingface/checkpoints/step-29610/"
    p2 = "/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/Transformer-Dispersion/pretrain_qwen3_0.6B_ckpts/step-29610-roundtrip/"
    compare(p1, p2)
