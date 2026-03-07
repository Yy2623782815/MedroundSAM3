PYTHONPATH=/root/autodl-tmp/work/sam3_med_agent_eval:/root/autodl-tmp/repos/sam3 \
python3 /root/autodl-tmp/work/sam3_med_agent_eval/eval_medical_agent_batch.py \
  --data_root /root/autodl-tmp/data/SAM3_data \
  --dataset AMOS2022 \
  --split test \
  --max_samples 4 \
  --max_agent_rounds 5 \
  --debug \
  --output_dir /root/autodl-tmp/work/sam3_med_agent_eval/outputs/AMOS2022_test50