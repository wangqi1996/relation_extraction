import os
import json
from collections import defaultdict
if __name__ == '__main__':
    with open('analysis.txt','w') as fout:
        outputs = []
        for i in range(10):
            fname = os.path.join('fold-{}'.format(i),'test.errors')
            with open(fname,'r') as f:
                a = json.load(f)
                d = dict()
                d['num_of_errors'] = len(a)
                d['seen2seen'] = len(list(filter(lambda x:x["gold_type"] == 'seen' and x['pred_type'] == 'seen',a)))
                d['seen2unseen'] = len(list(filter(lambda x:x["gold_type"] == 'seen' and x['pred_type'] == 'unseen',a)))
                d['unseen2seen'] = len(list(filter(lambda x:x["gold_type"] == 'unseen' and x['pred_type'] == 'seen',a)))
                d['unseen2unseen'] = len(list(filter(lambda x:x["gold_type"] == 'unseen' and x['pred_type'] == 'unseen',a)))

                # calculate the most five errors and count of their neighbours
                errors_attr = defaultdict(lambda:{})
                errors_cnt = defaultdict(lambda:0)
                for item in a:
                    errors_attr[item["pred"]] = {"count_of_neighbours":item["pred_neighbour_seen"] + item["pred_neighbour_unseen"],"type":item["pred_type"],"pred":item["pred"]}
                    errors_cnt[item["pred"]] += 1
                print(errors_cnt)
                top_five_pred = list(sorted(errors_cnt,key=errors_cnt.get,reverse=True))[:5]
                top_five_pred_attr = []
                for p in top_five_pred:
                    top_five_pred_attr.append(errors_attr["pred"])
                d["top_five_pred"] = top_five_pred_attr
                outputs.append(d)
        json.dump(outputs,fout,indent=4)

