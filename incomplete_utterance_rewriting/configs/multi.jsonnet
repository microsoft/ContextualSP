{
	"random_seed": 42,
	"numpy_seed": 42,
	"pytorch_seed": 42,
	"dataset_reader": {
		"type": "rewrite",
		"lazy": false,
		"super_mode": "before",
		"joint_encoding": true,
		"extra_stop_words": ["的", "是", "我", "了", "去"]
	},
	"train_data_path": "D:\\users\\v-qianl\\Unified-FollowUp\\dataset\\MultiDialogue\\train.txt",
	"validation_data_path": "D:\\users\\v-qianl\\Unified-FollowUp\\dataset\\MultiDialogue\\valid.txt",
	"model": {
		"type": "rewrite",
		"word_embedder": {
			"tokens": {
				"type": "embedding",
				"embedding_dim": 100,
				"trainable": true,
				"padding_index": 0
			}
		},
		"text_encoder": {
			"type": "lstm",
			"input_size": 100,
			"hidden_size": 200,
			"bidirectional": true,
			"num_layers": 1
		},
		"inp_drop_rate": 0.2,
		"out_drop_rate": 0.2,
		"feature_sel": 83,
		"loss_weights": [0.2, 0.2, 0.6],
		"super_mode": "before",
		"unet_down_channel": 64
	},
	"iterator": {
		"type": "basic",
		"batch_size": 16
	},
	"validation_iterator": {
		"type": "basic",
		"batch_size": 16
	},
	"trainer": {
		"num_epochs": 100,
		"cuda_device": 0,
		"patience": 10,
		"validation_metric": "+F3",
		"optimizer": {
			"type": "adam",
			"lr": 1e-3
		},
		"num_serialized_models_to_keep": 10,
		"should_log_learning_rate": true
	}
}