import time
import torch as t
import torch.nn as nn
import numpy as np
from torch import optim
from src.load_base import load_data, get_records
from src.evaluate import get_all_metrics
from sklearn.metrics import roc_auc_score, accuracy_score


class CKE(nn.Module):

    def __init__(self, n_entity, n_user, n_item, n_rels, dim):

        super(CKE, self).__init__()
        self.dim = dim
        user_emb_matrix = t.randn(n_user, dim)
        item_emb_matrix = t.randn(n_item, dim)
        ent_emb_matrix = t.randn(n_entity, dim)
        Mr_matrix = t.randn(n_rels, dim, dim)
        rel_emb_matrix = t.randn(n_rels, dim)
        nn.init.xavier_uniform_(user_emb_matrix)
        nn.init.xavier_uniform_(item_emb_matrix)
        nn.init.xavier_uniform_(ent_emb_matrix)
        nn.init.xavier_uniform_(Mr_matrix)
        nn.init.xavier_uniform_(rel_emb_matrix)
        self.user_emb_matrix = nn.Parameter(user_emb_matrix)
        self.item_emb_matrix = nn.Parameter(item_emb_matrix)
        self.ent_emb_matrix = nn.Parameter(ent_emb_matrix)
        self.Mr_matrix = nn.Parameter(Mr_matrix)
        self.rel_emb_matrix = nn.Parameter(rel_emb_matrix)

    def forward(self, data, name):
        if name == 'kg':
            # print(data)
            heads_id = [i[0] for i in data]
            relations_id = [i[1] for i in data]
            pos_tails_id = [i[2] for i in data]
            neg_tails_id = [i[3] for i in data]
            head_emb = self.ent_emb_matrix[heads_id].view(-1, 1, self.dim)
            rel_emb = self.rel_emb_matrix[relations_id].view(-1, 1, self.dim)
            pos_tail_emb = self.ent_emb_matrix[pos_tails_id].view(-1, 1, self.dim)
            neg_tail_emb = self.ent_emb_matrix[neg_tails_id].view(-1, 1, self.dim)
            Mr = self.Mr_matrix[relations_id]

            pos_stru_scores = (t.matmul(head_emb, Mr) + rel_emb - t.matmul(pos_tail_emb, Mr)).norm(dim=[1, 2]) ** 2
            neg_stru_scores = (t.matmul(head_emb, Mr) + rel_emb - t.matmul(neg_tail_emb, Mr)).norm(dim=[1, 2]) ** 2
            # print(t.log(t.sigmoid(pos_stru_scores - neg_stru_scores)))
            stru_loss = t.sigmoid(pos_stru_scores - neg_stru_scores)
            stru_loss = t.log(stru_loss).sum()
            return stru_loss
        else:

        # print(uvv)
            users_id = [i[0] for i in data]
            poss_id = [i[1] for i in data]
            negs_id = [i[2] for i in data]
            users_emb = self.user_emb_matrix[users_id]
            pos_items_emb = self.item_emb_matrix[poss_id] + self.ent_emb_matrix[poss_id]
            neg_items_emb = self.item_emb_matrix[negs_id] + self.ent_emb_matrix[negs_id]
            base_loss = t.sigmoid(t.mul(users_emb, pos_items_emb).sum(dim=1) - t.mul(users_emb, neg_items_emb).sum(dim=1))
            base_loss = t.log(base_loss).sum()

            return base_loss

    def get_predict(self, pairs):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        user_emb = self.user_emb_matrix[users]

        item_emb = self.item_emb_matrix[items] + self.ent_emb_matrix[items]
        score = (user_emb * item_emb).sum(dim=1)

        return score.cpu().detach().view(-1).numpy().tolist()


def get_scores(model, rec):
    scores = {}
    model.eval()
    for user in (rec):
        pairs = [[user, item] for item in rec[user]]
        predict = model.get_predict(pairs)
        # print(predict)
        n = len(pairs)
        user_scores = {rec[user][i]: predict[i] for i in range(n)}
        user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        scores[user] = user_list
    model.train()
    return scores


def eval_ctr(model, pairs, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model.get_predict(pairs[i: i+batch_size])
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np  = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = accuracy_score(true_label, pred_label)
    return round(auc, 3), round(acc, 3)


def get_uvvs(pairs):
    positive_dict = {}
    negative_dict = {}
    for pair in pairs:
        user = pair[0]
        item = pair[1]
        label = pair[2]
        if label == 1:
            if user not in positive_dict:
                positive_dict[user] = []

            positive_dict[user].append(item)
        else:
            if user not in negative_dict:
                negative_dict[user] = []

            negative_dict[user].append(item)
    data = []
    for user in positive_dict:
        size = len(positive_dict[user])
        # print(len(positive_dict[user]), len(negative_dict[user]))
        for i in range(size):
            pos_item = positive_dict[user][i]
            neg_item = negative_dict[user][i]
            data.append([user, pos_item, neg_item])

    np.random.shuffle(data)
    return data


def get_hrtts(kg_dict):
    # print('get hrtts...')

    entities = list(kg_dict)

    hrtts = []
    for head in kg_dict:
        for r_t in kg_dict[head]:
            relation = r_t[0]
            positive_tail = r_t[1]

            while True:
                negative_tail = np.random.choice(entities, 1)[0]
                if [relation, negative_tail] not in kg_dict[head]:
                    hrtts.append([head, relation, positive_tail, negative_tail])
                    break

    return hrtts


def train(args, is_topk=False):
    np.random.seed(555)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation =  data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    test_records = get_records(test_set)
    hrtts = get_hrtts(kg_dict)
    model = CKE(n_entity, n_user, n_item, n_relation, args.dim)

    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    uvvs = get_uvvs(train_set)
    train_data = [hrtts, uvvs]
    print(args.dataset + '----------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('learning_rate: %1.0e' % args.learning_rate, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    all_precision_list = []
    for epoch in range(args.epochs):

        start = time.clock()
        size = len(train_data[0])
        start_index = 0
        loss_sum = 0
        np.random.shuffle(train_set)
        np.random.shuffle(hrtts)
        while start_index < size:
            if start_index + args.batch_size <= size:
                hrtts = train_data[0][start_index: start_index + args.batch_size]
            else:
                hrtts = train_data[0][start_index:]
            loss = -model(hrtts, 'kg')
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        start_index = 0
        size = len(train_data[-1])
        while start_index < size:
            if start_index + args.batch_size <= size:
                uvvs = train_data[-1][start_index: start_index + args.batch_size]
            else:
                uvvs = train_data[-1][start_index:]
            loss = -model(uvvs, 'cf')
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        train_auc, train_acc = eval_ctr(model, train_set, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, args.batch_size)

        print('epoch: %d \t train_auc: %.3f \t train_acc: %.3f \t '
              'eval_auc: %.3f \t eval_acc: %.3f \t test_auc: %.3f \t test_acc: %.3f \t' %
              ((epoch+1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        precision_list = []
        if is_topk:
            scores = get_scores(model, rec)
            precision_list = get_all_metrics(scores, test_records)[0]
            print(precision_list, end='\t')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        all_precision_list.append(precision_list)
        end = time.clock()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.3f \t train_acc: %.3f \t eval_auc: %.3f \t eval_acc: %.3f \t '
          'test_auc: %.3f \t test_acc: %.3f \t' %
        (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
         test_auc_list[indices], test_acc_list[indices]), end='\t')

    print(all_precision_list[indices])

    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]


