#!/usr/bin/env python3
import json
import os
import sys
import torch

path_len = max(0, *map(len, sys.argv[1:]))
path_part = '{:' + str(path_len) + '}'

print((path_part + ' {:8} {:10} {:10} {:10}')
      .format('Path', 'n_epochs', 'best_epoch', 'train_err1', 'val_err1'))
for i in range(1, len(sys.argv)):
    try:
        with open(os.path.join(sys.argv[i], 'scores.tsv')) as f:
            names = f.readline().split()
            scores = [list(map(float, line.split())) for line in f]
        name2col = {n:i for i, n in enumerate(names)}
        scores = torch.Tensor(scores)
        argmax = scores.argmax(0)
        best_rol = scores[argmax[name2col['val_err1']], :]
        print((path_part + '{:8d} {:10d} {:10.2f} {:10.2f}')
              .format(sys.argv[i], scores.size(0),
                      int(best_rol[name2col['epoch']]),
                      best_rol[name2col['train_err1']],
                      best_rol[name2col['val_err1']],
                      ))
    except FileNotFoundError:
        pass
