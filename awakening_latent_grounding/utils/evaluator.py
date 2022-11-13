from utils.data_loader import Text2SQLDataset
import torch
from collections import OrderedDict, defaultdict
from contracts import *

class GroundingEvaluator:
    def __init__(self, dataset: Text2SQLDataset) -> None:
        self.dataset = dataset

        self.statistics = defaultdict(int)
        self.logs: List[str] = []
        self.dataset_logs = defaultdict(list)
        self.dataset_statistics = defaultdict(dict)

    def add_batch(self, model_inputs, model_outputs):
        batch_size = len(model_inputs['input_token_ids'])
        self.statistics['total_count'] += batch_size
        self.statistics['loss'] += model_outputs['loss'].item() * batch_size
        self.statistics['cp_loss'] += model_outputs['cp_loss'].item() * batch_size
        if 'grounding_loss' in model_outputs:
            self.statistics['grounding_loss'] += model_outputs['grounding_loss'].item() * batch_size
        if 'sequence_labeling_loss' in model_outputs:
            self.statistics['sequence_labeling_loss'] += model_outputs['sequence_labeling_loss'].item() * batch_size

        if not 'grounding_scores' in model_outputs:
            model_outputs['grounding_scores'] = torch.zeros(model_inputs['concept_labels'].size(0), model_inputs['concept_labels'].size(1), model_inputs['question_mask'].size(1))

        for idx in range(batch_size):
            example: Text2SQLExample = self.dataset.get_example_by_id(model_inputs['id'][idx].item())['example']
            outputs = {
                'concept_scores': model_outputs['concept_scores'][idx], 
                'grounding_scores': model_outputs['grounding_scores'][idx]
            }
            if 'question_label_scores' in model_outputs:
                outputs['question_label_scores'] = model_outputs['question_label_scores'][idx]
            self._add_one(
                example=example,
                inputs={ key : val[idx] for key, val in model_inputs.items() if isinstance(val, list) or isinstance(val, torch.Tensor) },
                outputs=outputs,
            )

    def post_process_inputs_and_outputs(self, example: Text2SQLExample, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
        question_length = len(example.question.tokens)

        mask_size = len(inputs['concept_labels']) - len(All_Agg_Op_Keywords) - example.schema.num_columns - len(example.matched_values)
        sizes = [len(All_Agg_Op_Keywords), example.schema.num_columns, len(example.matched_values), mask_size]
        split_concept_labels = torch.split(inputs['concept_labels'], sizes)
        split_concept_scores = torch.split(outputs['concept_scores'], sizes)
        inputs['concept_labels'] = {
            SQLTokenType.Keyword.string: split_concept_labels[0],
            SQLTokenType.Column.string: split_concept_labels[1],
            SQLTokenType.Value.string: split_concept_labels[2]
        }

        outputs["concept_scores"] = {
            SQLTokenType.Keyword.string: split_concept_scores[0],
            SQLTokenType.Column.string: split_concept_scores[1],
            SQLTokenType.Value.string: split_concept_scores[2]
        }

        split_grounding_scores = torch.split(outputs['grounding_scores'], sizes)
        outputs["grounding_scores"] = {
            SQLTokenType.Keyword.string: split_grounding_scores[0][:, :question_length],
            SQLTokenType.Column.string: split_grounding_scores[1][:, :question_length],
            SQLTokenType.Value.string: split_grounding_scores[2][:, :question_length]
        }

        return inputs, outputs


    def _add_one(self, example: Text2SQLExample, inputs: Dict, outputs: Dict) -> None:
        def _get_concept(concept_type, index) -> Concept:
            if concept_type == SQLTokenType.Column.string:
                return example.schema.columns[index]
            if concept_type == SQLTokenType.Table.string:
                return example.schema.tables[index]
            if concept_type == SQLTokenType.Value.string:
                return example.matched_values[index]
            # if concept_type == SQLTokenType.Function.string:
            #     column_idx, agg_idx = index // len(Aggregation), index % len(Aggregation)
            #     return DependentConcept(base_concept=example.schema.columns[column_idx], name=str(Aggregation(agg_idx)))
            # if concept_type == SQLTokenType.Operator.string:
            #     value_idx, op_idx = index // len(ComparisonOperator), index % len(ComparisonOperator)
            #     return DependentConcept(base_concept=example.matched_values[value_idx], name=str(ComparisonOperator(op_idx)))
            if concept_type == SQLTokenType.Keyword.string:
                return Keyword(keyword=All_Agg_Op_Keywords[index].upper())
            raise NotImplementedError(concept_type)

        def tensor_to_list(data):
            ans = [num.item() for num in data]
            return ans
        
        def get_question_label_logs(example: Text2SQLExample, gold_labels: List[int], pred_labels: List[int], is_correct: bool):
            question_text = example.question.text
            tokens = example.question.tokens

            gold_labels = [QuestionLabel.get_abbr_by_value(tag) for tag in gold_labels]
            pred_labels = [QuestionLabel.get_abbr_by_value(tag) for tag in pred_labels]

            label_logs = []

            label_logs += ['\nQuestion: {}\n'.format(question_text)]
            label_logs += ['Dataset: {}; Schema: {}\n'.format(example.dataset, example.schema.to_string())]
            label_logs += ['SQL: {}\n'.format(str(example.sql))]
            
            gold_strs = ['{}/{: <3}'.format(token.token, tag) for (token, tag) in zip(tokens, gold_labels)]
            pred_strs = ['{}/{: <3}'.format(token.token, tag) for (token, tag) in zip(tokens, pred_labels)]
            label_logs += ['Gold: {}\n'.format(' '.join(gold_strs))]
            label_logs += ['Pred: {}\n'.format(' '.join(pred_strs))]
            label_logs += ['Sequence Label : {}\n\n'.format(is_correct)]

            return label_logs
        
        question_length = len(example.question.tokens)
        inputs, outputs = self.post_process_inputs_and_outputs(example, inputs, outputs)

        log_start_idx = len(self.logs)
        self.logs += ['Question: {}\n'.format(example.question.text)]
        self.logs += ['Dataset: {}; Schema: {}\n'.format(example.dataset, example.schema.to_string())]
        self.logs += ['SQL: {}\n'.format(str(example.sql))]

        input_tokens = self.dataset.get_example_by_id(inputs['id'].item())['input_tokens']
        self.logs += ['Input Tokens: {}\n'.format(" ".join([x for x in input_tokens]))]
        concept_types = inputs['concept_labels'].keys()
        cp_results = {}

        if example.dataset not in self.dataset_statistics:
            self.dataset_statistics[example.dataset] = defaultdict(int)
        
        self.dataset_statistics[example.dataset]['total_count'] += 1

        if 'question_label_scores' in outputs:
            # pick valid part
            gold_labels = inputs['question_labels'][:question_length]
            pred_labels = torch.argmax(outputs['question_label_scores'], dim=-1)[:question_length]
            
            is_correct = bool(gold_labels.equal(pred_labels)) if len(gold_labels) > 0 else True
            self.statistics['Question Sequence True'] += int(is_correct)
            self.statistics['Question Sequence All'] += 1

            # log question labeling result
            question_label_logs = get_question_label_logs(example, gold_labels, pred_labels, is_correct)
            self.logs.extend(question_label_logs)


            for label in QuestionLabel.get_all_labels():
                label_value = label.value
                label_type = QuestionLabel.get_abbr_by_value(label_value)
                # calculate every question label type

                pred_true_list = pred_labels.eq(gold_labels)
                this_label_pos = gold_labels.eq(label_value)

                if any(this_label_pos):
                    key_true = f'{label_type} tag True'
                    key_all = f'{label_type} tag All'

                    self.statistics[key_true] += sum(pred_true_list & this_label_pos).item()
                    self.statistics[key_all] += sum(this_label_pos).item()

                    self.dataset_statistics[example.dataset][key_true] += sum(pred_true_list & this_label_pos).item()
                    self.dataset_statistics[example.dataset][key_all] += sum(this_label_pos).item()

        for concept_type in concept_types:
            gold_labels = inputs['concept_labels'][concept_type]
            pred_labels = (outputs['concept_scores'][concept_type] >= 0.5).to(torch.long)
            is_correct = bool(gold_labels.equal(pred_labels)) if len(gold_labels) > 0 else True
            self.statistics["{} CP True".format(str(concept_type))] += int(is_correct)
            self.statistics["{} CP All".format(str(concept_type))] += 1
            cp_results[concept_type] = is_correct
            self.dataset_statistics[example.dataset]["{} CP True".format(str(concept_type))] += int(is_correct)
            self.dataset_statistics[example.dataset]["{} CP All".format(str(concept_type))] += 1

            for i in range(len(gold_labels)):
                if gold_labels[i] == 0 and pred_labels[i] == 0:
                    continue
                concept = _get_concept(concept_type, i)
                self.logs.append("{}: {}, Gold = {}; Pred = {}/{:.4f}, Correct = {}\n".format(
                    str(concept_type), concept.identifier, gold_labels[i].item(), pred_labels[i].item(), outputs['concept_scores'][concept_type][i].item(), gold_labels[i].item() == pred_labels[i].item()))

                grounding_vector = outputs['grounding_scores'][concept_type][i].cpu().tolist()
                grounding_strs = ["{}/{:.3f}".format(token, g_score) for g_score, token in zip(grounding_vector, example.question.text_tokens)]
                self.logs.append("Grounding: {}\n".format(" ".join(grounding_strs)))

        self.logs.append("Concept: {}\n".format("; ".join(["{} = {}".format(str(key), bool(val)) for key, val in cp_results.items()])))
        self.logs.append("\n")

        self.dataset_logs[example.dataset] += self.logs[log_start_idx:]

    def get_metrics(self, log_saved_file: str=None) -> Dict:
        metrics = OrderedDict()
        total_count = self.statistics['total_count']
        metrics['loss'] = self.statistics['loss'] / total_count
        metrics['cp_loss'] = self.statistics['cp_loss'] / total_count
        if 'grounding_loss' in self.statistics:
            metrics['grounding_loss'] = self.statistics['grounding_loss'] / total_count
        if 'sequence_labeling_loss' in self.statistics:
            metrics['sequence_labeling_loss'] = self.statistics['sequence_labeling_loss'] / total_count

        for label in QuestionLabel.get_all_labels():
            label_value = label.value
            label_type = QuestionLabel.get_abbr_by_value(label_value)

            key_true = f'{label_type} tag True'
            key_all = f'{label_type} tag All'

            if key_true in self.statistics:
                key_tag_accuracy = f'[{label_type}] accuracy'
                metrics[key_tag_accuracy] = self.statistics[key_true] / self.statistics[key_all]

            if len(self.dataset_statistics) > 1:
                for dataset in self.dataset_statistics.keys():
                    statistics = self.dataset_statistics[dataset]
                    if key_true in statistics:
                        accuracy = statistics[key_true] / statistics[key_all]
                        key_dataset_tag_accuracy = f'[{label_type}] {dataset} accuracy'
                        metrics[key_dataset_tag_accuracy] = accuracy
        
        for concept_type in [SQLTokenType.Keyword.string, SQLTokenType.Column.string, SQLTokenType.Value.string]:
            if '{} CP True'.format(str(concept_type)) in self.statistics:
                metrics['{} accuracy'.format(str(concept_type))] = self.statistics['{} CP True'.format(str(concept_type))] / self.statistics['{} CP All'.format(str(concept_type))]

            if len(self.dataset_statistics) > 1:
                for dataset in self.dataset_statistics.keys():
                    statistics = self.dataset_statistics[dataset]
                    if '{} CP True'.format(str(concept_type)) in statistics:
                        accuracy = statistics['{} CP True'.format(str(concept_type))] / statistics['{} CP All'.format(str(concept_type))]
                        metrics['{} {} accuracy'.format(str(concept_type), dataset)] = accuracy

        if log_saved_file is not None:
            with open(log_saved_file, 'w', encoding='utf-8') as fw:
                for key, val in metrics.items():
                    fw.write("{}\t{:.4f}\n".format(key, val))
                fw.write('\n')
                fw.writelines(self.logs)

            if len(self.dataset_statistics) > 1:
                for dataset in self.dataset_statistics.keys():
                    with open(log_saved_file.replace(".txt", ".{}.txt".format(dataset)), 'w', encoding='utf-8') as fw:
                        for key, val in metrics.items():
                            if key.startswith(dataset):
                                fw.write("{}\t{:.4f}\n".format(key, val))
                        fw.write('\n')
                        fw.writelines(self.dataset_logs[dataset])

        return metrics

