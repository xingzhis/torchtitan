import os
from datasets import load_dataset                                                                                                                                                                                 

os.makedirs('./local_datasets', exist_ok=True)

print('Downloading EleutherAI/fineweb-edu-dedup-10b...')                                                                                                                                                          
ds = load_dataset('EleutherAI/fineweb-edu-dedup-10b', split='train', num_proc=16)  # Match cpus-per-task in sbatch                                                                                                                                        

# NO NEED TO SHUFFLE. it is already shuffled, and it takes forever to shuffle.
# print(f'Downloaded {len(ds)} samples')                                                                                                                                                                            
# print('Shuffling with seed=42...')                                                                                                                                                                                
# ds_shuffled = ds.shuffle(seed=42)                                                                                                                                                                                 
                                                                                                                                                                                                                
print('Saving to disk...')                                                                                                                                                                                        
ds_shuffled.save_to_disk('./local_datasets/fineweb_edu_10B_shuffled')                                                                                                                                             
print('Done!')  