#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import torch
from torch.autograd import Variable
import numpy as np


def batch_to_cuda(batch, volatile=False):
    # moves dataset batch on GPU

    q = Variable(batch[0], requires_grad=False).cuda()
    a = Variable(batch[1], requires_grad=False).cuda()
    n_votes = Variable(batch[2], requires_grad=False).cuda()
    i = Variable(batch[4], requires_grad=False).cuda()
    k = Variable(batch[5], requires_grad=False).cuda()
    qlen = Variable(batch[6], requires_grad=False).cuda()
    return q, a, n_votes, i, k, qlen


def save(model, optimizer, ep, epoch_loss, epoch_acc, dir, name):
    # saves model and optimizer state

    tbs = {
        'epoch': ep + 1,
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    torch.save(tbs, os.path.join(dir, name + '.pth.tar'))


def total_vqa_score(output_batch, n_votes_batch):
    # computes the total vqa score as assessed by the challenge

    vqa_score = 0
    _, oix = output_batch.data.max(1)
    for i, pred in enumerate(oix):
        count = n_votes_batch[i,pred]
        vqa_score += min(count.cpu().data.item()/3, 1)
    return vqa_score

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)
