# -*- coding: utf-8 -*-
"""
S4PRED batched runner: loads the model once, runs vectorized inference in minibatches,
prints/saves outputs in the same formats as the original script.
"""

from __future__ import print_function
import sys
sys.path.append("s4pred")

import argparse, os
import numpy as np
import torch

from network import S4PRED
from utilities import loadfasta

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(
    description='Predict Secondary Structure with the S4PRED model',
    epilog='Takes a FASTA file containing an arbitrary number of sequences and outputs a prediction for each.'
)
parser.add_argument('input', metavar='input', type=str,
                    help='FASTA file (can contain many sequences).')
parser.add_argument('-d','--device', metavar='d', type=str, default='cpu',
                    help="Device: 'cpu', 'gpu' (alias for cuda:0), or 'cuda:N' (default: cpu).")
parser.add_argument('-t','--outfmt', metavar='m', type=str, default='ss2',
                    help='Output format: ss2, fas, or horiz (default: ss2).')
parser.add_argument('-c','--fas-conf', default=False, action='store_true',
                    help='Include confidence scores in .fas output.')
parser.add_argument('-s','--silent', default=False, action='store_true',
                    help='Suppress printing predictions to stdout.')
parser.add_argument('-z','--save-files', default=False, action='store_true',
                    help='Save each input sequence prediction to a file.')
parser.add_argument('-o','--outdir', metavar='p', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                    help='Directory where files are saved if --save-files is used.')
parser.add_argument('-x','--save-by-idx', default=False, action='store_true',
                    help='If saving, name files by a counter instead of sequence ID.')
parser.add_argument('-t2','--outfmt2', metavar='n', type=str, default='',
                    help='Optional second output format: ss2, fas, or horiz.')
parser.add_argument('-p','--prefix', metavar='n', type=str, default=None,
                    help='Use this prefix for output filenames.')
parser.add_argument('-T','--threads', metavar='n', type=int, default=None,
                    help='Number of CPU threads for PyTorch (default: all cores).')

# NEW:
parser.add_argument('--batch-size', type=int, default=128,
                    help='Sequences per minibatch (default: 128).')
parser.add_argument('--amp', action='store_true',
                    help='Use CUDA autocast (mixed precision).')

args = parser.parse_args()
args_dict = vars(args)

# ----------------------------
# Device & threads
# ----------------------------
if args_dict['threads']:
    torch.set_num_threads(args_dict['threads'])

dev_arg = args_dict['device'].lower()
if dev_arg.startswith('gpu'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
elif dev_arg.startswith('cuda'):
    device = torch.device(dev_arg if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# ----------------------------
# Model (load ONCE)
# ----------------------------
s4pred = S4PRED().to(device)
s4pred.eval()
s4pred.requires_grad_(False)

scriptdir = os.path.dirname(os.path.realpath(__file__))
weight_files = [
    '/weights/weights_1.pt',
    '/weights/weights_2.pt',
    '/weights/weights_3.pt',
    '/weights/weights_4.pt',
    '/weights/weights_5.pt'
]
# Load ensemble weights (as in your original)
s4pred.model_1.load_state_dict(torch.load(scriptdir + weight_files[0], map_location='cpu'))
s4pred.model_2.load_state_dict(torch.load(scriptdir + weight_files[1], map_location='cpu'))
s4pred.model_3.load_state_dict(torch.load(scriptdir + weight_files[2], map_location='cpu'))
s4pred.model_4.load_state_dict(torch.load(scriptdir + weight_files[3], map_location='cpu'))
s4pred.model_5.load_state_dict(torch.load(scriptdir + weight_files[4], map_location='cpu'))
# (models are already on device via S4PRED().to(device); state_dict tensors move with module parameters)

# Optional CUDA speedups
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')  # allow TF32 on Ampere/Ada
    # You can also set env: TORCH_ALLOW_TF32=1 externally

# ----------------------------
# Data
# ----------------------------
# loadfasta returns, per sequence: data = (id, indices_int_array, aa_string)
seqs = loadfasta(args_dict['input'])

# ----------------------------
# Output helpers (unchanged)
# ----------------------------
ind2char = {0: "C", 1: "H", 2: "E"}

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def format_ss2(data, ss, ss_conf):
    lines = ['# PSIPRED VFORMAT (S4PRED V1.2.4)\n']
    for i in range(len(ss)):
        lines.append("%4d %c %c  %6.3f %6.3f %6.3f" % (
            i + 1, data[2][i], ind2char[ss[i]], ss_conf[i,0], ss_conf[i,1], ss_conf[i,2]))
    return lines

def format_fas(data, ss, ss_conf, include_conf=False):
    lines = ['>' + data[0]]
    lines.append(data[2])
    lines.append("".join([ind2char[int(s)] for s in ss]))
    if include_conf:
        lines.append(np.array2string(ss_conf[:,0], max_line_width=10**9, precision=3,
                                     formatter={'float_kind':lambda x: "%.3f" % x}).replace('[','').replace(']',''))
        lines.append(np.array2string(ss_conf[:,1], max_line_width=10**9, precision=3,
                                     formatter={'float_kind':lambda x: "%.3f" % x}).replace('[','').replace(']',''))
        lines.append(np.array2string(ss_conf[:,2], max_line_width=10**9, precision=3,
                                     formatter={'float_kind':lambda x: "%.3f" % x}).replace('[','').replace(']',''))
    return lines

def format_horiz(data, ss, ss_conf):
    lines = ['# PSIPRED HFORMAT (S4PRED V1.2.4)']
    sub_seqs = list(chunkstring(data[2], 60))
    sub_ss   = list(chunkstring("".join([ind2char[int(s)] for s in ss]), 60))
    num_len  = int(np.floor(len(data[2]) / 10))
    num_seq  = ''.join(f'{str((i+1)*10):>10}' for i in range(num_len+1))
    num_seq  = list(chunkstring(num_seq, 60))
    conf_idxs = ss_conf.argmax(-1)
    confs = ss_conf[np.arange(len(conf_idxs)), conf_idxs[:]]
    confs = "".join([str(x) for x in np.floor(confs * 10).astype(np.int32)])
    confs = list(chunkstring(confs, 60))
    for idx, subsq in enumerate(sub_seqs):
        lines.append(f'\nConf: {confs[idx]}')
        lines.append(f'Pred: {sub_ss[idx]}')
        lines.append(f'  AA: {subsq}')
        lines.append(f'      {num_seq[idx]}\n')
    return lines

# ----------------------------
# Batched prediction
# ----------------------------
def _predict_batch(batch_datas):
    """
    batch_datas: list of data tuples from loadfasta: (id, int_indices, aa_string)
    Returns: list of (ss (np.int64[L]), ss_conf (np.float32[L,3])) in the same order
    """
    # lengths and padding
    ints = [torch.as_tensor(d[1], dtype=torch.long) for d in batch_datas]
    lengths = [t.numel() for t in ints]
    Lmax = max(lengths)
    B = len(ints)

    x = torch.full((B, Lmax), fill_value=0, dtype=torch.long)  # assuming 0 is a valid pad (model should ignore/pad)
    for i, t in enumerate(ints):
        x[i, :t.numel()] = t
    x = x.to(device)

    # forward
    with torch.inference_mode():
        if device.type == 'cuda' and args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = s4pred(x)  # expected shape [B, Lmax, 3] in log space
        else:
            logits = s4pred(x)

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            
        # move out of log space and (re)normalize for numerical hygiene
        conf = logits.exp()
        conf = conf / conf.sum(dim=-1, keepdim=True)

        ss = conf.argmax(dim=-1)  # [B, Lmax]

    # slice back per-length and to numpy
    out = []
    conf = conf.detach().cpu().numpy()
    ss = ss.detach().cpu().numpy()
    for i, L in enumerate(lengths):
        out.append( (ss[i, :L].astype(np.int64), conf[i, :L, :].astype(np.float32)) )
    return out

# ----------------------------
# Save/print control
# ----------------------------
output_dir = args_dict['outdir']
if output_dir[-1] != '/':
    output_dir += '/'

scriptdir = os.path.dirname(os.path.realpath(__file__)) + '/'
if output_dir == scriptdir and args_dict['save_files']:
    os.makedirs(scriptdir + 'preds/', exist_ok=True)
    output_dir += 'preds/'

bs = max(1, int(args.batch_size))
N = len(seqs)

# Process in minibatches, but model stays loaded once
for start in range(0, N, bs):
    chunk = seqs[start:start+bs]
    preds = _predict_batch(chunk)  # list of (ss, ss_conf) aligned to chunk

    for j, (ss, ss_conf) in enumerate(preds):
        data = chunk[j]  # (id, ints, aastring)
        if args_dict['outfmt'] == 'ss2':
            lines = format_ss2(data, ss, ss_conf); suffix = '.ss2'
        elif args_dict['outfmt'] == 'fas':
            lines = format_fas(data, ss, ss_conf, include_conf=args_dict['fas_conf']); suffix = '.fas'
        elif args_dict['outfmt'] == 'horiz':
            lines = format_horiz(data, ss, ss_conf); suffix = '.horiz'
        else:
            raise ValueError('Invalid output format. Use ss2, fas, or horiz.')

        if not args_dict['silent']:
            try:
                for line in lines:
                    print(line)
            except BrokenPipeError:
                pass
        else:
            if not args_dict['save_files']:
                raise ValueError('Using --silent without --save-files yields no output.')

        if args_dict['save_files']:
            if args_dict['save_by_idx']:
                file_name = f's4_out_{start + j}{suffix}'
            else:
                file_name = (args_dict['prefix'] + suffix) if args_dict['prefix'] else (data[0] + suffix)
            file_path = output_dir + file_name
            with open(file_path, 'w') as f:
                for line in lines:
                    f.write(line + '\n')

    # Optional second format
    if len(args_dict['outfmt2']) > 2:
        for j, (ss, ss_conf) in enumerate(preds):
            data = chunk[j]
            if args_dict['outfmt2'] == 'ss2':
                lines = format_ss2(data, ss, ss_conf); suffix = '.ss2'
            elif args_dict['outfmt2'] == 'fas':
                lines = format_fas(data, ss, ss_conf, include_conf=args_dict['fas_conf']); suffix = '.fas'
            elif args_dict['outfmt2'] == 'horiz':
                lines = format_horiz(data, ss, ss_conf); suffix = '.horiz'
            else:
                raise ValueError('Invalid 2nd output file format.')
            if not args_dict['silent']:
                for line in lines:
                    print(line)
            if args_dict['save_files']:
                if args_dict['save_by_idx']:
                    file_name = f's4_out_{start + j}{suffix}'
                else:
                    file_name = (args_dict['prefix'] + suffix) if args_dict['prefix'] else (data[0] + suffix)
                file_path = output_dir + file_name
                with open(file_path, 'w') as f:
                    for line in lines:
                        f.write(line + '\n')
