{
	"random_seed": 42,
	"numpy_seed": 42,
	"pytorch_seed": 42,
	"dataset_reader": {
		"type": "rewrite",
		"lazy": false,
		"super_mode": "before",
		"joint_encoding": true,
		"extra_stop_words": [
			"of",
			"about",
			"the",
			"any",
			"for"
		]
	},
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
			"hidden_size": 300,
			"bidirectional": true,
			"num_layers": 1
		},
		"inp_drop_rate": 0.1,
		"out_drop_rate": 0.1,
		"feature_sel": 115,
		"loss_weights": [
			0.3,
			0.3,
			0.4
		],
		"super_mode": "before",
		"enable_training_log": true,
		"unet_down_channel": 128
	},
	"iterator": {
		"type": "basic",
		"batch_size": 12
	},
	"validation_iterator": {
		"type": "basic",
		"batch_size": 12
	},
	"trainer": {
		"num_epochs": 100,
		"cuda_device": 0,
		"patience": 10,
		"validation_metric": "+EM",
		"optimizer": {
			"type": "adam",
			"lr": 2e-4,
			"weight_decay": 1e-5
		},
		"num_serialized_models_to_keep": 10,
		"should_log_learning_rate": true
	}
}