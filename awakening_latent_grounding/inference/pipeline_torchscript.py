"""
Model prediction implementation based on TorchScript
"""
import os
import torch
from .pipeline_base import NLBindingInferencePipeline

class NLBindingTorchScriptPipeline(NLBindingInferencePipeline):
    def __init__(self,
        model_dir: str,
        greedy_linking: bool,
        threshold: float=0.2,
        num_threads: int=8,
        use_gpu: bool = torch.cuda.is_available()
    ) -> None:
        super().__init__(model_dir, greedy_linking=greedy_linking, threshold=threshold)

        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        model_file = "nl_binding.script.bin"

        torch.set_num_interop_threads(2)
        torch.set_num_threads(num_threads)
        print('Torch model Threads: {}, {}, {}'.format(torch.get_num_interop_threads(), torch.get_num_threads(), self.device))
        model_ckpt_path = os.path.join(model_dir, model_file)
        self.model = torch.jit.load(model_ckpt_path, map_location=self.device)
        self.model.eval()

    def run_model(self, **inputs):
        input_token_ids = torch.as_tensor(inputs['input_token_ids'], dtype=torch.long, device=self.device)
        entity_indices = torch.as_tensor(inputs['entity_indices'], dtype=torch.long, device=self.device)
        question_indices = torch.as_tensor(inputs['question_indices'], dtype=torch.long, device=self.device)

        cp_scores, grounding_scores = self.model(input_token_ids, entity_indices, question_indices)
        cp_scores = cp_scores.cpu().tolist()
        grounding_scores = grounding_scores.cpu().tolist()

        if 'spm_idx_mappings' in inputs:
            idx_mappings = inputs['spm_idx_mappings']
            raw_grounding_scores = [[-100 for _ in idx_mappings] for _ in grounding_scores]
            for k, _ in enumerate(grounding_scores):
                for idx1, idx2 in enumerate(idx_mappings):
                    if idx1 > 0 and idx_mappings[idx1 - 1] == idx_mappings[idx1]:
                        continue
                    raw_grounding_scores[k][idx1] = grounding_scores[k][idx2]
            grounding_scores = raw_grounding_scores

        return {
            'cp_scores': cp_scores,
            'grounding_scores': grounding_scores
        }
