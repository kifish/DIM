
import numpy as np
test_out_filename = "persona_test_out.txt"

def to_sessions(results):
    sessions = []
    one_sess = []
    i = 0
    for prob,label in results:
        i += 1
        one_sess.append((prob, int(label)))
        if i % 20 == 0:
            one_sess_np = np.array(one_sess)
            if one_sess_np[:, 1].sum() == 1:
                sessions.append(one_sess)
            else:
                print('this session has no positive example or this'
                'session has more than one example')
            one_sess = []
    assert len(sessions) == results / 20
    return sessions




def recall_at_position_k(sort_data, k):
    sort_label = [s_d[1] for s_d in sort_data]
    select_label = sort_label[:k]
    return 1.0 * select_label.count(1) / sort_label.count(1)

def evaluation_one_session(data):
    '''
    :param data: one conversion session(actually it means one turn), which layout is [(score1, label1), (score2, label2), ..., (score20, label20)].
    :return: all kinds of metrics used in paper.
    '''
    np.random.shuffle(data)
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    r_1 = recall_at_position_k(sort_data, 1)
    r_2 = recall_at_position_k(sort_data, 2)
    r_5 = recall_at_position_k(sort_data, 5)
    return r_1, r_2, r_5





# prob,label
with open(test_out_filename, 'r') as f:
    results = []
    for line in f.readlines():
        prob, label = line.strip().split('\t')
        prob = float(prob)
        label = float(label)
        results.append(prob,label)
    print('the number of examples : {}'.format(results)) # 19个负例和1个正例算20个
    print('the number of positive examples : {}'.format(results / 20 ))
    sessions = to_sessions(results)
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0
    for session in sessions:
        r_1, r_2, r_5 = evaluation_one_session(session)
        sum_r_1 += r_1
        sum_r_2 += r_2
        sum_r_5 += r_5

    total_s = len(sessions)
    recall = {}
    recall["recall@1"] = sum_r_1/total_s
    recall["recall@2"] = sum_r_2/total_s
    recall["recall@5"] = sum_r_5/total_s

    print("num_query = {}".format(total_s))
    print("recall@1 = {}".format(recall["recall@1"]))
    print("recall@2 = {}".format(recall["recall@2"]))
    print("recall@5 = {}".format(recall["recall@5"]))

