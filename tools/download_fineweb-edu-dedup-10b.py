import os
from datasets import load_dataset                                                                                                                                                                                 

os.makedirs('./local_datasets', exist_ok=True)

print('Downloading EleutherAI/fineweb-edu-dedup-10b...')                                                                                                                                                          
ds = load_dataset('EleutherAI/fineweb-edu-dedup-10b', split='train')                                                                                                                                              
                                                                                                                                                                                                                
print(f'Downloaded {len(ds)} samples')                                                                                                                                                                            
print('Shuffling with seed=42...')                                                                                                                                                                                
ds_shuffled = ds.shuffle(seed=42)                                                                                                                                                                                 
                                                                                                                                                                                                                
print('Saving to disk...')                                                                                                                                                                                        
ds_shuffled.save_to_disk('./local_datasets/fineweb_edu_10B_shuffled')                                                                                                                                             
print('Done!')  