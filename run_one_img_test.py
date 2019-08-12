import torch
import numpy as np
from spacy.tokenizer import Tokenizer

import en_core_web_sm
import os
from complete_graph_model import Model
try:
    import cPickle as pickle
except:
    import pickle as pickle





def init_pretrained_wemb(emb_dim):  # move this inside model init?
    embeddings_index = {}
    f = open(os.path.join(data_dir, 'glove.6B.') +
             str(emb_dim) + 'd.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype=np.float32)
        embeddings_index[word] = coefs
    f.close()

    embedding_mat = np.zeros((q_words, emb_dim), dtype=np.float32)
    for word, i in q_wtoi.items():
        embedding_v = embeddings_index.get(word)
        if embedding_v is not None:
            embedding_mat[i] = embedding_v
    return embedding_mat




def run_model_one_img_test(model, img_feat, question):
    nlp = en_core_web_sm.load()
    tokenizer = Tokenizer(nlp.vocab)

    question_tokens = [t.text if '?' not in t.text else t.text[:-1]
                       for t in tokenizer(question.lower())]
    question_tokens_idx = [q_wtoi[w] for w in question_tokens]

    res = model(
        torch.tensor([np.pad(np.asarray(question_tokens_idx), (0, 100-len(question_tokens)), 'constant')], dtype=torch.int64).cuda(),
        torch.tensor([img_feat], dtype=torch.float32).cuda(),
        torch.tensor([img_feat.shape[0]], dtype=torch.int64).cuda(),
        [torch.tensor(data=4,dtype=torch.int64)]
        )

    return res


if __name__ == '__main__':
    im_dir = '/home/anastasia/DATA/vqa-2/val2014'
    features_dir = '/home/anastasia/DATA/rcnn/val2014'
    model_file = '/home/anastasia/vqa-graph/save-1/model_13.pth.tar'
    data_dir = 'data'
    im_name = 'COCO_val2014_000000066485.jpg'

    q_dict = pickle.load(
        open(os.path.join(data_dir, 'train_q_dict.p'), 'rb'))
    q_itow = q_dict['itow']
    q_wtoi = q_dict['wtoi']
    q_words = len(q_itow) + 1

    pretrained_wemb_300 = init_pretrained_wemb(300)

    model = Model(vocab_size=15186,
                  emb_dim=300,
                  feat_dim=2052,
                  hid_dim=1024,
                  out_dim=3001,
                  dropout=0.5,
                  pretrained_wemb=pretrained_wemb_300,
                  neighbourhood_size=8,
                  return_attention=True
                  )

    # move to CUDA
    model = model.cuda()

    # Restore pre-trained model
    ckpt = torch.load(model_file)
    model.load_state_dict(ckpt['state_dict'])

    img_feat_file = os.path.join(features_dir, im_name.replace('.jpg', '.npy'))
    img_dic = np.load(img_feat_file, encoding='latin1', allow_pickle=True).item()
    bboxes = img_dic['image_bboxes']

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        bbox[0] /= img_dic['image_w']
        bbox[1] /= img_dic['image_h']
        bbox[2] /= img_dic['image_w']
        bbox[3] /= img_dic['image_h']
        bboxes[i] = bbox

    img_feat = np.concatenate([img_dic['image_feat'], bboxes], axis=1)

    question = 'how many people here'

    res = run_model_one_img_test(model, img_feat, question)

    print(res)
