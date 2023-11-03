# OPTIONAL:
import sys
import pickle
import json
import torch
import copy
from genre.trie import Trie, MarisaTrie
from genre.fairseq_model import mGENRE
from custom_model import *


model = mGENRE.from_pretrained("model/finetune_cl_with_mgenre").cuda().eval()
lang_name = sys.argv[1]
# file_path 为待预测文件路径
file_path = "test_data/mantra_v2/gener_type_list_Patent_" + lang_name + ".txt"
#file_path = sys.argv[1]
wo = open("multi_lingual_predict_result/" + lang_name, "w")
#wo = open("analysis_badcase.txt", "w")


num = 0
num_acc = 0
with open(file_path) as f:
    for line in f:
        num += 1
        json_data = json.loads(line.strip())
        mention_context = json_data["mention_context"]
        if len(mention_context) > 500:
            e_s = mention_context.find("[START]") - 50
            e_s = e_s if e_s > 0 else 0
            e_n= mention_context.find("[END]") + 50
            mention_context = mention_context[e_s: e_n]
            
        golden_cuis = set(json_data["golden_cui"].split("|"))
        cand = json_data["cand"]
        cand_items = cand.keys()
        trie_of_mention = Trie([[2] + model.encode(cand_text)[1:].tolist() for cand_text in cand_items])
        sentences = [mention_context]
        result = model.sample(
                            sentences,
                            7,
                            #prefix_allowed_tokens_fn=lambda batch_id, sent: [
                            #e for e in trie_of_mention.get(sent.tolist())
                            #if e < len(model.task.target_dictionary)
                            #],
               )

        # 选出预测结果列表
        pred_cui = []
        score_tensor = []
        pred_text = []
        for res_item in result[0]:
            if '<unk>' not in res_item["text"] and res_item["text"] in cand:
                for cur_cui_id in cand[res_item["text"]]:
                    score_tensor.append(res_item['score'])
                    pred_cui.append(cur_cui_id)
                    pred_text.append(res_item["text"])

        if not pred_cui:
            wo.write(json.dumps({}) + "\n")
            continue

        # score归一化
        #score_tensor = torch.tensor(score_tensor)
        #score_tensor = torch.nn.functional.softmax(score_tensor)
        for i in range(len(pred_cui)):
            pred_cui[i] = {"cui_id": pred_cui[i], "text": pred_text[i], "score": float(score_tensor[i])}

        pred_cui_list = sorted(pred_cui, key=lambda x: x["score"], reverse=True)
        pred_cuis = pred_cui_list[0]["cui_id"].split("|")
        json_data["pred"] = pred_cui_list
        json_data["pred_cui"] = pred_cuis
        if len(set(pred_cuis) & golden_cuis) > 0:
            num_acc += 1
        wo.write(json.dumps(json_data, ensure_ascii=False) + "\n")
        
wo.close()
