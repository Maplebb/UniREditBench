import os, glob
from safetensors.torch import load_file, save_file

in_dir  = "./ckpt"
out_file = "./ckpt/model.safetensors"

state = {}
for fp in sorted(glob.glob(os.path.join(in_dir, "model*.safetensors"))):
    if os.path.basename(fp).endswith("model.safetensors"):
        continue
    part = load_file(fp) 
    state.update(part)

save_file(state, out_file)
