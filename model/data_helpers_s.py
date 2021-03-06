#encoding=utf8
import numpy as np
import random


def load_vocab(fname):
    '''
    vocab = {"I": 0, ...}
    '''
    vocab={}
    with open(fname, 'rt') as f:
        for i,line in enumerate(f):
            word = line.decode('utf-8').strip()
            vocab[word] = i
    return vocab

def load_char_vocab(fname):
    '''
    charVocab = {"U": 0, "!": 1, ...}
    '''
    charVocab={}
    with open(fname, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            char_id = int(fields[0])
            ch = fields[1]
            charVocab[ch] = char_id
    return charVocab

def to_vec(tokens, vocab, maxlen):
    '''
    length: length of the input sequence
    vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
    '''
    n = len(tokens)
    length = 0
    vec=[]
    for i in range(n):
        length += 1
        if tokens[i] in vocab:
            vec.append(vocab[tokens[i]])
        else:
            vec.append(vocab["fiance"])  # fix to fiance
    return length, np.array(vec)

def load_dataset_s(fname, vocab, max_utter_num, max_utter_len, max_response_len, max_persona_len):
    dataset=[]
    with open(fname, 'rt') as f:
        for us_id, line in enumerate(f):
            line = line.decode('utf-8').strip()
            fields = line.split('\t')
            # context utterances
            context = fields[0]
            utterances = (context + " ").split(' _eos_ ')[:-1]
            utterances = [utterance + " _eos_" for utterance in utterances]
            utterances = utterances[-max_utter_num:]   # select the last max_utter_num utterances
            us_tokens = []
            us_vec = []
            us_len = []
            for utterance in utterances:
                u_tokens = utterance.split(' ')[:max_utter_len]  # select the head max_utter_len tokens in every utterance
                u_len, u_vec = to_vec(u_tokens, vocab, max_utter_len)
                us_tokens.append(u_tokens)
                us_vec.append(u_vec)
                us_len.append(u_len)
            us_num = len(utterances)
            
            # response
            response = fields[1]
            rs_tokens = []
            rs_vec = []
            rs_len = []
            r_tokens = response.split(' ')[:max_response_len]  # select the head max_response_len tokens in every candidate
            r_len, r_vec = to_vec(r_tokens, vocab, max_response_len)
            rs_tokens.append(r_tokens) 
            rs_vec.append(r_vec)
            rs_len.append(r_len)

            # label
            label = float(fields[2])

            # other persona
            if fields[3] != "NA" and fields[4] == "NA":
                personas = fields[3].split("|")
                ps_tokens = []
                ps_vec = []
                ps_len = []
                for persona in personas:
                    p_tokens = persona.split(' ')[:max_persona_len]  # select the head max_persona_len tokens in every persona
                    p_len, p_vec = to_vec(p_tokens, vocab, max_persona_len)
                    ps_tokens.append(p_tokens)
                    ps_vec.append(p_vec)
                    ps_len.append(p_len)
                ps_num = len(personas)

            # self persona
            if fields[3] == "NA" and fields[4] != "NA":
                personas = fields[4].split("|")
                ps_tokens = []
                ps_vec = []
                ps_len = []
                for persona in personas:
                    p_tokens = persona.split(' ')[:max_persona_len]  # select the head max_persona_len tokens in every persona
                    p_len, p_vec = to_vec(p_tokens, vocab, max_persona_len)
                    ps_tokens.append(p_tokens)
                    ps_vec.append(p_vec)
                    ps_len.append(p_len)
                ps_num = len(personas)

            dataset.append((us_id, us_tokens, us_vec, us_len, us_num, rs_tokens, rs_vec, rs_len, label, ps_tokens, ps_vec, ps_len, ps_num))
   
    return dataset

def normalize_vec(vec, maxlen):
    '''
    pad the original vec to the same maxlen
    [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
    '''
    if len(vec) == maxlen:
        return vec

    new_vec = np.zeros(maxlen, dtype='int32')
    for i in range(len(vec)):
        new_vec[i] = vec[i]
    return new_vec


def charVec(tokens, charVocab, maxlen, maxWordLength):
    '''
    chars = np.array( (maxlen, maxWordLength) )    0 if not found in charVocab or None
    word_lengths = np.array( maxlen )              1 if None
    '''
    n = len(tokens)
    if n > maxlen:
        n = maxlen

    chars =  np.zeros((maxlen, maxWordLength), dtype=np.int32)
    word_lengths = np.ones(maxlen, dtype=np.int32)
    for i in range(n):
        token = tokens[i][:maxWordLength]
        word_lengths[i] = len(token)
        row = chars[i]
        for idx, ch in enumerate(token):
            if ch in charVocab:
                row[idx] = charVocab[ch]

    return chars, word_lengths



def batch_iter_s(data, batch_size, num_epochs, max_utter_num, max_utter_len, max_response_num, max_response_len, 
                max_persona_num, max_persona_len, charVocab, max_word_length, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            # 避免batch的shape 不一致
            if (batch_num + 1) * batch_size > data_size:
                continue
            x_utterances = []
            x_utterances_len = []

            x_responses = []
            x_responses_len = []

            x_labels = []
            x_ids = []
            x_utterances_num = []

            x_utterances_char=[]
            x_utterances_char_len=[]

            x_responses_char=[]
            x_responses_char_len=[]

            x_personas = []
            x_personas_len = []
            x_personas_char=[]
            x_personas_char_len=[]
            x_personas_num = []

            for rowIdx in range(start_index, end_index):
                us_id, us_tokens, us_vec, us_len, us_num, rs_tokens, rs_vec, rs_len, label, ps_tokens, ps_vec, ps_len, ps_num = data[rowIdx]

                # normalize us_vec and us_len
                new_utters_vec = np.zeros((max_utter_num, max_utter_len), dtype='int32')
                new_utters_len = np.zeros((max_utter_num, ), dtype='int32')
                for i in range(len(us_len)):
                    new_utter_vec = normalize_vec(us_vec[i], max_utter_len)
                    new_utters_vec[i] = new_utter_vec
                    new_utters_len[i] = us_len[i]
                x_utterances.append(new_utters_vec)
                x_utterances_len.append(new_utters_len)

                # rs_tokens, rs_vec, rs_len
                # normalize rs_vec and rs_len
                new_responses_vec = np.zeros((max_response_num, max_response_len), dtype='int32')
                new_responses_len = np.zeros((max_response_num, ), dtype='int32')
                for i in range(len(rs_len)):
                    new_response_vec = normalize_vec(rs_vec[i], max_response_len)
                    new_responses_vec[i] = new_response_vec
                    new_responses_len[i] = rs_len[i]
                    break
                # fake data
                tmp_x_response = []
                tmp_x_response_len = []
                for _ in range(20):
                    tmp_x_response.append(new_responses_vec[0])
                    tmp_x_response_len.append(new_responses_len[0])
                x_responses.append(tmp_x_response)
                x_responses_len.append(tmp_x_response_len)

                x_labels.append(label)

                x_ids.append(us_id)
                x_utterances_num.append(us_num)

                # normalize us_CharVec and us_CharLen
                uttersCharVec = np.zeros((max_utter_num, max_utter_len, max_word_length), dtype='int32')
                uttersCharLen = np.ones((max_utter_num, max_utter_len), dtype='int32')
                for i in range(len(us_len)):
                    utterCharVec, utterCharLen = charVec(us_tokens[i], charVocab, max_utter_len, max_word_length)
                    uttersCharVec[i] = utterCharVec
                    uttersCharLen[i] = utterCharLen
                x_utterances_char.append(uttersCharVec)
                x_utterances_char_len.append(uttersCharLen)

                # normalize rs_CharVec and rs_CharLen
                rsCharVec = np.zeros((max_response_num, max_response_len, max_word_length), dtype='int32')
                rsCharLen = np.ones((max_response_num, max_response_len), dtype='int32')
                for i in range(20):
                    rCharVec, rCharLen = charVec(rs_tokens[i], charVocab, max_response_len, max_word_length)
                    rsCharVec[i] = rCharVec
                    rsCharLen[i] = rCharLen
                    break
                # fake data
                tmp_response_char = []
                tmp_responses_char_len = []
                for _ in range(20):
                    tmp_response_char.append(rsCharVec[0])
                    tmp_responses_char_len.append(rsCharLen[0])
                x_responses_char.append(tmp_response_char)
                x_responses_char_len.append(tmp_responses_char_len)


                # normalize ps_vec and ps_len
                new_personas_vec = np.zeros((max_persona_num, max_persona_len), dtype='int32')
                new_personas_len = np.zeros((max_persona_num, ), dtype='int32')
                for i in range(len(ps_len)):
                    new_persona_vec = normalize_vec(ps_vec[i], max_persona_len)
                    new_personas_vec[i] = new_persona_vec
                    new_personas_len[i] = ps_len[i]
                x_personas.append(new_personas_vec)
                x_personas_len.append(new_personas_len)

                # normalize ps_CharVec and ps_CharLen
                psCharVec = np.zeros((max_persona_num, max_persona_len, max_word_length), dtype='int32')
                psCharLen = np.ones((max_persona_num, max_persona_len), dtype='int32')
                for i in range(len(ps_len)):
                    pCharVec, pCharLen = charVec(ps_tokens[i], charVocab, max_persona_len, max_word_length)
                    psCharVec[i] = pCharVec
                    psCharLen[i] = pCharLen
                x_personas_char.append(psCharVec)
                x_personas_char_len.append(psCharLen)

                x_personas_num.append(ps_num)


            # debug 
            # print('x_responses : {}'.format(np.array(x_responses).shape))
            # print('x_responses_len : {}'.format(np.array(x_responses_len).shape))
            # print('x_labels : {}'.format(np.array(x_labels).shape))
            # print('x_responses_char : {}'.format(np.array(x_responses_char).shape))
            # print(np.array(x_responses_char))
            # print('x_responses_char_len : {}'.format(np.array(x_responses_char_len).shape))

            yield np.array(x_utterances), np.array(x_utterances_len), np.array(x_responses), np.array(x_responses_len), \
                  np.array(x_utterances_num), np.array(x_labels), x_ids, \
                  np.array(x_utterances_char), np.array(x_utterances_char_len), np.array(x_responses_char), np.array(x_responses_char_len), \
                  np.array(x_personas), np.array(x_personas_len), np.array(x_personas_char), np.array(x_personas_char_len), np.array(x_personas_num)



if __name__ == '__main__':
    import tensorflow as tf
    # Files
    tf.flags.DEFINE_string("train_file", "../data/personachat_s_processed/processed_train_self_original.txt", "path to train file")
    tf.flags.DEFINE_string("valid_file", "../data/personachat_s_processed/processed_valid_self_original.txt", "path to valid file")
    tf.flags.DEFINE_string("vocab_file", "../data/personachat_s_processed/vocab.txt", "vocabulary file")
    tf.flags.DEFINE_string("char_vocab_file",  "../data/personachat_s_processed/char_vocab.txt", "path to char vocab file")
    tf.flags.DEFINE_string("embedded_vector_file", "../data/personachat_s_processed/glove_42B_300d_vec_plus_word2vec_100.txt", "pre-trained embedded word vector")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("max_utter_num", 15, "max utterance number")
    tf.flags.DEFINE_integer("max_utter_len", 20, "max utterance length")
    tf.flags.DEFINE_integer("max_response_num", 20, "max response candidate number")
    tf.flags.DEFINE_integer("max_response_len", 20, "max response length")
    tf.flags.DEFINE_integer("max_persona_num", 5, "max persona number")
    tf.flags.DEFINE_integer("max_persona_len", 15, "max persona length")
    tf.flags.DEFINE_integer("max_word_length", 18, "max word length")
    tf.flags.DEFINE_integer("embedding_dim", 400, "dimensionality of word embedding")
    tf.flags.DEFINE_integer("rnn_size", 200, "number of RNN units")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 40, "batch size (default: 128)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "dropout keep probability (default: 1.0)")
    tf.flags.DEFINE_integer("num_epochs", 10, "number of training epochs (default: 1000000)")
    tf.flags.DEFINE_integer("evaluate_every", 500, "evaluate model on valid dataset after this many steps (default: 1000)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    vocab = load_vocab(FLAGS.vocab_file)
    print('vocabulary size: {}'.format(len(vocab)))
    charVocab = load_char_vocab(FLAGS.char_vocab_file)
    print('charVocab size: {}'.format(len(charVocab)))
    train_dataset = load_dataset_s(FLAGS.train_file, vocab, FLAGS.max_utter_num, FLAGS.max_utter_len, FLAGS.max_response_len, FLAGS.max_persona_len)
    print('train dataset size: {}'.format(len(train_dataset)))

    print('building dataset...')
    batches = batch_iter_s(train_dataset, FLAGS.batch_size, FLAGS.num_epochs, FLAGS.max_utter_num, FLAGS.max_utter_len, \
                                        FLAGS.max_response_num, FLAGS.max_response_len, FLAGS.max_persona_num, FLAGS.max_persona_len, \
                                        charVocab, FLAGS.max_word_length, shuffle=False) # 20个不分开
    print('shape:')
    for batch in batches:
        print('----------')
        break
    print('dataset builded...')

    '''
    vocabulary size: 20879
    charVocab size: 69
    train dataset size: 1314380
    building dataset...
    dataset builded...
    shape:
    x_responses : (40, 20, 20)
    x_responses_len : (40, 20)
    x_labels : (40,)
    x_responses_char : (40, 20, 20, 18)
    x_responses_char_len : (40, 20, 20)
    # dim.r_charVec: x_r_char, 即 x_responses_char
    '''


