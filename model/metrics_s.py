#encoding=utf8
import math


def group_results(results):
    group = []
    ret = []
    for line in results:
        group.append(line)
        if len(group) == 20:
            ret.append(group)
            group = []
    if len(group) == 20:
        ret.append(group)
    return ret


def is_valid_query(group):
    num_pos = 0
    num_neg = 0
    for prob, label in group:
        if label  > 0:
            num_pos += 1
        else:
            num_neg += 1
    if num_pos > 0 and num_pos == 1 and num_neg > 0: # 如果算map可以考虑取消这个num_pos == 1的限制
        return True
    else: # 只有一类样本，return false
        return False

def get_num_valid_query(results):
    groups = group_results(results)
    num_query = sum(map(is_valid_query,groups))
    return num_query

def top_1_precision(results): # r1
    groups = group_results(results)
    num_query = sum(map(is_valid_query,groups))
    top_1_correct = 0.0
    for group in groups:
        if not is_valid_query(group):
            continue
        sorted_group = sorted(group, key=lambda x : x[0], reverse=True)
        prob, label = sorted_group[0]
        if label > 0:
            top_1_correct += 1

    if num_query > 0:
        return top_1_correct/num_query
    else:
        return 0.0

def mean_reciprocal_rank(results):
    groups = group_results(results)
    num_query = sum(map(is_valid_query,groups))
    mrr = 0.0
    for group in groups:
        if not is_valid_query(group):
            continue
        sorted_group = sorted(group, key=lambda x : x[0], reverse=True)
        for i,(prob,label) in enumerate(sorted_group):
            if label >  0:
                mrr += 1.0/(i+1)
                break

    if num_query == 0:
        return 0.0
    else:
        mrr = mrr/num_query
        return mrr

def mean_average_precision(results):
    groups = group_results(results)
    num_query = sum(map(is_valid_query,groups))
    mvp = 0.0
    for group in groups:
        if not is_valid_query(group):
            continue
        sorted_group = sorted(group, key=lambda x : x[0], reverse=True)
        num_relevant_doc = 0.0
        avp = 0.0
        for i,(prob,label) in enumerate(sorted_group):
            if label == 1:
                num_relevant_doc += 1
                precision = num_relevant_doc/(i+1)
                avp += precision
        avp = avp/num_relevant_doc
        mvp += avp

    if num_query == 0:
        return 0.0
    else:
        mvp = mvp/num_query
        return mvp



def classification_metrics(results):
    total_num = 0
    total_correct = 0
    true_positive = 0
    positive_correct = 0
    predicted_positive = 0

    loss = 0.0;
    for prob, label in results: 
        # label, 0 or 1
        # score : float (0,1)
        total_num += 1
        if prob > 0.5:
            predicted_positive += 1

        if label > 0:
            true_positive += 1
            loss += -math.log(prob+1e-12)
        else:
            loss += -math.log(1.0 - prob + 1e-12);
        if prob > 0.5 and label > 0:
            total_correct += 1
            positive_correct += 1 
        if prob < 0.5 and label < 0.5:
            total_correct += 1

    accuracy = float(total_correct)/total_num
    precision = float(positive_correct)/(predicted_positive+1e-12)
    recall    = float(positive_correct)/true_positive # R1 < recall < R20
    F1 = 2.0 * precision * recall/(1e-12+precision + recall)
    return accuracy, precision, recall, F1, loss/total_num;
