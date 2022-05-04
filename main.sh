# python main.py  --backbone resnet50 \
#                 --ytvos_path data \
#                 --masks \
#                 --dim_feedforward 512

python main.py --backbone resnet50 \
               --ytvos_path data \
               --masks \
               --pretrained_weights ./models/pretrained/vistr_r50.pth \
               --output_dir ./checkpoint/resnet50_e1_d1/
