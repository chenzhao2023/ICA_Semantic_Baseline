This project uses tree-lstm for coronary artery semantic labeling.

using environment pytorch1.10 ())



### new information Aug 2023

AAAI 2024

1. prepare data

data_prepare_aaai.py

- parser.add_argument('--base_path', type=str, default='/media/z/data2/artery_semantic_segmentation')
- parser.add_argument('--data_path', type=str, default="/media/z/data2/artery_semantic_segmentation/hnn_hm_june/data/artery_with_feature")
- parser.add_argument('--save_path', type=str, default="data_aai")
- parser.add_argument('--project_path', type=str, default="pytorch-tree-lstm")
- parser.add_argument('--image_size', type=int, default=512)

2. train model

command: 

python tree_lstm_artery_aaai.py --exp=exp_aaai --data_file_path=data_aaai --model=bi --in_features=12 --mlp_hidden=128 --lstm_hidden=32 --num_class=5 --device=cuda:0 --train=train --cv=0

python tree_lstm_artery_aaai.py --exp=exp_aaai --data_file_path=data_aaai --model=bi --in_features=12 --mlp_hidden=128 --lstm_hidden=32 --num_class=5 --device=cuda:0 --train=train --cv=1

python tree_lstm_artery_aaai.py --exp=exp_aaai --data_file_path=data_aaai --model=bi --in_features=12 --mlp_hidden=128 --lstm_hidden=32 --num_class=5 --device=cuda:0 --train=train --cv=2

python tree_lstm_artery_aaai.py --exp=exp_aaai --data_file_path=data_aaai --model=bi --in_features=12 --mlp_hidden=128 --lstm_hidden=32 --num_class=5 --device=cuda:1 --train=train --cv=3

python tree_lstm_artery_aaai.py --exp=exp_aaai --data_file_path=data_aaai --model=bi --in_features=12 --mlp_hidden=128 --lstm_hidden=32 --num_class=5 --device=cuda:1 --train=train --cv=4