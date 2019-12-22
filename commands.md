
<!-- FROZEN MODEL TO TENSORFLOWJS MODEL -->
tensorflowjs_converter \
  --input_format=tf_frozen_model \
  --output_node_names=final_result \
  tf_files/quantized_graph.pb \
  tf_files/web

<!-- RETRAIN MODEL -->
python retrain-mobilenet-v2/examples/image_retraining/retrain.py \
  --image_dir=tf_files/dataset \
  --tfhub_module=$MODULE \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --bottleneck_dir=tf_files/bottlenecks \
  --summaries_dir=tf_files/training_summaries \
  --intermediate_output_graphs_dir=tf_files/intermediate_graphs/ \
  --intermediate_store_frequency=500 \
  --saved_model_dir=tf_files/saved_model \
  --how_many_training_steps=2000 \
  --learning_rate=0.0333

<!-- TEST YOUR MODEL -->
  python scripts/label_image.py \
  --graph=tf_files/retrained_graph.pb \
  --input_width=$IMAGE_SIZE \
  --input_height=$IMAGE_SIZE \
  --image=tf_files/dataset/Baroque/File-2.jpg

<!-- QUANTIZES MODEL -->
  python scripts/quantize_graph.py \
  --input=tf_files/retrained_graph.pb \
  --output=tf_files/quantized_graph.pb \
  --output_node_names=final_result \
  --mode=weights_rounded