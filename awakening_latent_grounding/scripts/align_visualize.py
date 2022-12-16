#%%
import sys
sys.path.append("..")
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
from models import *
from utils import *
from datetime import datetime
device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')
#%%
# Generate a custom diverging colormap
def format_text(text: str) -> str:
    return r"\textit{" + text.replace("_", "\_") + "}"

cmap = sn.diverging_palette(230, 20, as_cmap=True)
plt.rc('text', usetex=True)
def visualize_alignment(align_matrix: np.ndarray, question: List[str], entities: List[str], saved_path: str=None):
    entities = [e.replace("_", "\_") for e in entities]
    with sn.axes_style("white"):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax = sn.heatmap(
            align_matrix,
            fmt=".2f",
            vmax=0.5,
            center=0,
            cmap=cmap,
            square=True,
            cbar=None,
            annot=True,
            #cbar_kws=False, #{"orientation": "horizontal"},
            xticklabels=question, 
            yticklabels=entities)
        
        ax.set_yticklabels(entities, rotation=0, fontsize="12", va="center")
        ax.set_xticklabels(question, rotation=45, fontsize="12")
        plt.tight_layout()

        #plt.rc('xtick', labelsize=12) 
        #plt.rc('ytick', labelsize=12) 
        #plt.rc('text', usetex=True)
        # ax.set_title('SELECT semesters.semester_name , semesters.semester_id FROM semesters , student_enrolment ON semesters.semester_id = student_enrolment.semester_id GROUP BY semesters.semester_id ORDER BY COUNT ( * ) DESC LIMIT 1')
        if saved_path is not None:
            plt.savefig(saved_path)
        plt.show()
    pass

#%%
question = ['show', 'name', ',', 'country', ',', 'age', 'for', 'all', 'singers', 'ordered', 'by', 'age', 'from', 'the', 'oldest', 'to', 'the', 'youngest', '.']
columns = ['singer', 'singer.name', 'singer.country', 'singer.age']
align_matrix = np.array(
    [[0.000,0.010 ,0.002,0.015 ,0.000,0.001,0.000,0.000,0.967,0.000,0.000,0.002,0.000,0.000,0.001,0.000,0.000,0.000,0.000],
    [0.013,0.369 ,0.053,0.194 ,0.001,0.001,0.002,0.003,0.359,0.001,0.001,0.001,0.000,0.001,0.000,0.001,0.001,0.000,0.001],
    [0.001,0.017 ,0.045,0.888 ,0.002,0.007,0.001,0.001,0.027,0.001,0.000,0.004,0.000,0.000,0.002,0.001,0.000,0.002,0.000],
    [0.000,0.001 ,0.000,0.003 ,0.001,0.313,0.000,0.000,0.005,0.001,0.001,0.353,0.000,0.000,0.215,0.000,0.000,0.105,0.000]
])

visualize_alignment(align_matrix, question, columns)
# %%
def load_model_and_data_iter(ckpt_path, data_path):
    config = json.load(open(os.path.join(os.path.dirname(ckpt_path), 'config.json'), 'r', encoding='utf-8'))
    config['checkpoint'] = ckpt_path
    config['device'] = device

    model = load_model_from_checkpoint(**config)
    model.eval()
    print('-------------------Config-------------------')
    for key, val in config.items():
        print(key, val)
    print('load {} from {} over .'.format(config['model'], ckpt_path))
    
    bert_version = config['bert_version']
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    examples = json.load(open(data_path, 'r', encoding='utf-8'))

    dataset = SpiderDataset(examples, tokenizer, device, 512, False)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tensor_collate_fn(x, False))

    data = {}
    for batch_input in data_loader:
        question = batch_input['example'][0]['question']['text']
        data[question] = batch_input
    return model, data

model, data = load_model_and_data_iter(
    r'/home/deyang/NLBinding/Experiments/pt/spider_alignment_202101101250/spider_alignment/self-learning_202101101252/student.step_26000.acc_0.738.f1_0.879.pt',
    r'/home/deyang/NLBinding/Experiments/data/slsql/dev.bert-large-uncased-whole-word-masking.json')

#%%
def predict(question: str):
    assert question in data, question
    model_input = data[question]
    with torch.no_grad():
        model_output = model(**model_input)
        align_matrix: torch.Tensor = model_output['alignment_weights'][0]
        tbl_scores = torch.softmax(model_output['table_logits'][0], dim=-1)
        col_scores = torch.softmax(model_output['column_logits'][0], dim=-1)
        schema: SpiderSchema = SpiderSchema.from_json(model_input['example'][0]['schema'])
        question: Utterance = Utterance.from_json(model_input['example'][0]['question'])
        print(schema.to_string())

        entities, align_vectors = [], []
        for t_idx in range(schema.num_tables):
            assert t_idx < len(schema.column_names_original), t_idx
            entity_name = schema.get_tbl_identifier_name(t_idx)
            if tbl_scores[t_idx][1].item() < 0.5: # and align_matrix[t_idx].max().item() < 0.2:
                continue
            entities.append(entity_name)
            align_vectors.append(align_matrix[t_idx])
        for c_idx in range(schema.num_columns):
            entity_name = schema.get_col_identifier_name(c_idx)
            if col_scores[c_idx][1].item() < 0.5: #and align_matrix[c_idx + schema.num_tables].max().item() < 0.2:
                continue
            entities.append(entity_name)
            align_vectors.append(align_matrix[c_idx + schema.num_tables])
        
        align_matrix = torch.stack(align_vectors, dim=0).cpu().numpy()
        saved_path = os.path.join('/home/deyang/NLBinding/Experiments/scripts/pdfs', 'spider_{}.pdf'.format(datetime.now().strftime("%Y%m%d%H%M")))
        visualize_alignment(align_matrix, question.text_tokens, entities, saved_path)
# %%
predict('Show name, country, age for all singers ordered by age from the oldest to the youngest.')
#%%
predict('Where is the youngest teacher from?')
# %%
predict('For each semester, what is the name and id of the one with the most students registered?')
# %%
