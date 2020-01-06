from collections import defaultdict


def micro_precision(pred, gold):
    correct = 0
    for p, g in zip(pred, gold):
        if p == g:
            correct += 1
    return correct * 1.0 / len(gold)


def each_precision(pred, gold):
    gold_relation_length = defaultdict(lambda: 0)
    tp = defaultdict(lambda: 0)
    for p, g in zip(pred, gold):
        gold_relation_length[g] += 1
        if p == g:
            tp[g] += 1
    each_precision = []
    for g in gold_relation_length:
        each_precision.append(tp[g] * 1.0 / gold_relation_length[g])
    return each_precision


def macro_precision(pred, gold):
    precision = each_precision(pred, gold)
    return sum(precision) / len(precision)


def cal_micro_macro_all(pred, gold, seens, unseens):
    unseen_result = []
    seen_result = []
    all_result = []
    all_num = len(pred)
    seen_acc = 0
    unseen_acc = 0

    for p, g in zip(pred, gold):
        if g in seens:
            seen_result.append(str(g) + "\t" + str(p))
            if p == g:
                seen_acc += 1
        elif g in unseens:
            unseen_result.append(str(g) + '\t' + str(p))

            if p == g:
                unseen_acc += 1
        else:
            assert False, u"tian,处理又有问题"

        all_result.append(str(g) + '\t' + str(p))

    seen_macro = cal_macro_acc(seen_result)
    unseen_macro = cal_macro_acc(unseen_result)
    all_macro = cal_macro_acc(all_result)

    seen_micro = seen_acc / len(seen_result)
    unseen_micro = unseen_acc / len(unseen_result)
    all_micro = (seen_acc + unseen_acc) / all_num

    return seen_macro, unseen_macro, all_macro, seen_micro, unseen_micro, all_micro


def cal_macro_acc(output):
    '''
    对每一个关系计算准确率，然后准确率相加算关系的平均准确率
    :param output: str gold_relation\t predict_relation
    :return: relation score
    '''
    rel_map = {}
    rel_correct = {}

    for t in output:
        r = t.split("\t")
        gold = r[0]
        pre = r[1]
        if gold in rel_map:
            rel_map[gold] += 1
        else:
            rel_map[gold] = 1
        # 如果预测的等于真实值
        if pre == gold:
            if gold in rel_correct:
                rel_correct[gold] += 1
            else:
                rel_correct[gold] = 1
    true_relation_score = 0
    for rel in rel_map:
        all_num = rel_map[rel]
        if rel in rel_correct:
            true_num = rel_correct[rel]
            true_relation_score += true_num * 1.0 / all_num
    if len(rel_map) == 0:
        return 0
    true_relation_score /= len(rel_map)
    return true_relation_score
