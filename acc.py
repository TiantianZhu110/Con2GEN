import json
import re
import sys
import copy


def acc_top1_top5():
    file_path = sys.argv[1]
    num = 0 
    num_acc_top1 = 0
    num_acc_top5 = 0
    with open(file_path) as f:
        for line in f:
            num += 1
            json_data = json.loads(line.strip())
            if "golden_cui" not in json_data:
                continue
            golden_cuis = set(json_data["golden_cui"].split("|"))
            pred_cuis = set(json_data["pred_cui"])
            if len(golden_cuis & pred_cuis) > 0:
                num_acc_top1 += 1

            # top5
            pred_res = json_data["pred"]
            top5_cands = pred_res[0: 5]
            top5_cuis = []
            if "cui_id" not in pred_res[0]:
                text_id = json_data["cand"]
                for item in top5_cands:
                    text = item["text"]
                    if text in text_id:
                        top5_cuis.extend(text_id[text].strip().split("|"))
            else:
                for item in top5_cands:
                    top5_cuis.extend(item["cui_id"].split("|"))
            if len(set(top5_cuis) & golden_cuis) > 0:
                num_acc_top5 += 1
    print("num: %d, num_acc_top1: %d, acc_top1: %f" % (num, num_acc_top1, float(num_acc_top1) / num))
    print("num: %d, num_acc_top5: %d, acc_top5: %f" % (num, num_acc_top5, float(num_acc_top5) / num))



if __name__ == "__main__":
    acc_top1_top5()

