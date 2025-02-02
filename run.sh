export PYTHONPATH=./
export model=bert_gnn_lstm
export dataset=recreation_dataset

if [ -d "./data/$dataset/processed" ]; then
  rm -r ./data/$dataset/processed
  echo "remove the dir ./data/$dataset/processed"
fi

export num_mid_layers=4
export num_heads=8
export threshold=3


CUDA_VISIBLE_DEVICES=0 python main.py \
--config_path ./src/model/config/conf_$model.ini \
--data_path ./data/$dataset \
--epoch 40 --train_batch_size 8 \
--num_mid_layers $num_mid_layers \
--num_heads $num_heads \
--threshold $threshold \
--eval_frequency 2 \
--save_model_name models/Model-ExtractionNet-layer-num-$num_mid_layers-heads-$num_heads-threshold-$threshold-dataset-$dataset.ckpt
