
run_cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_inpainting_model.py"

echo ${run_cmd}
eval ${run_cmd}
