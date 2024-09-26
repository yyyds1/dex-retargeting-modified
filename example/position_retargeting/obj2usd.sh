# python obj2usd.py \
#  input /home/hz/Downloads/dataset/models/002_master_chef_can/textured_simple.obj \
#  output /home/hz/code/dex-retargeting-modified/example/position_retargeting/export_usd \
#  --mass 0.1
# conda init bash
# conda activate isaaclab
python obj2usd.py \
 /home/hz/Downloads/dataset/models/002_master_chef_can/textured.obj \
 /home/hz/code/dex-retargeting-modified/example/position_retargeting/export_usd/textured_simple \
 --mass 0.1
 --make-instanceable True