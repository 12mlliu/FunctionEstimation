# Deep Learning for function estimation


## Data generation

first generate data, and export it into a dataset that can be used for training. This can easily be done by setting `--export_data true`:
```bash
python main.py --export_data true --dump_path dataset/ --batch_size 32 --cpu true --exp_name firsttry --num_workers 1 --env_base_seed -1  --n_variables 1 --leaf_probs "0.75,0.25,0" --max_ops 5 --max_int 5 --positive true --max_len 64 --epoch_size 10 --operators "add:10,sub:3,mul:10,pow2:4,pow3:2,pow4:1,pow5:1,exp:4,sin:4,cos:4,tan:4"

```
Data will be exported in the prefix and infix formats to:
- `./dumped/exp_name/EXP_ID/data.prefix`
- `./dumped/exp_name/EXP_ID/data.infix`

`data.prefix` and `data.infix` are two parallel files containing the same number of lines, with the same equations written in prefix and infix representations respectively. In these files, each line contains an input (e.g. the function to integrate) and the associated output (e.g. an integral) separated by a tab. In practice, the model only operates on prefix data. The infix data is optional, but more human readable, and can be used for debugging purposes.

If you generate your own dataset, you will notice that the generator generates a lot of duplicates (which is inevitable if you parallelize the generation). In practice, we remove duplicates using:

[bug]
```bash
cat ./dataset/firsttry/*/data.prefix | awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' > data.prefix.counts

```

The resulting format is the following:
```
count1|input1_prefix    data_x,data_y
count2|input2_prefix    data_x,data_y
...
```
[bug]
Where the input and output are separated by a tab, and equations are sorted by counts. This is under this format that data has to be given to the model. The number of `counts` is not used by the model, but was not removed in case of potential curriculum learning. The last part consists in simply splitting the dataset into training / validation / test sets. This can be done with the `split_data.py` script:

```bash
# create a valid and a test set of 10k equations
python split_data.py data.prefix.counts 10000

# remove valid inputs that are in the train
mv data.prefix.counts.valid data.prefix.counts.valid.old
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' <(cat data.prefix.counts.train) data.prefix.counts.valid.old \
> data.prefix.counts.valid

# test test inputs that are in the train
mv data.prefix.counts.test data.prefix.counts.test.old
awk -F"[|\t]" 'NR==FNR { lines[$2]=1; next } !($2 in lines)' <(cat data.prefix.counts.train) data.prefix.counts.test.old \
> data.prefix.counts.test
```


## Training

```bash

python main.py --exp_name first_train --reload_data "data.prefix.counts.train,data.prefix.counts.valid,data.prefix.counts.test" --reload_size 30 --emb_dim 8 --n_enc_layers 6 --n_dec_layers 6 --n_heads 2 --optimizer "adam,lr=0.0001"  --batch_size 30 --epoch_size 2 --validation_metrics valid_prim_fwd_acc --cpu true


```

[bug] --fp16 true --amp 2     # float16 training
[bug] --validation_metrics valid_prim_fwd_acc 



