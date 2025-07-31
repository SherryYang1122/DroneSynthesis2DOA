# evaluation

## Usage

To evaluate, the following files are necessary:

- DOA estimation result folder (csv format)
- Real DOA files (csv format)

These data or file can be generated from other subprojects which are saved in **Data/exp..** automatically.

An example of how to run the script to evaluate is as follows:

```
python eval_runner_main.py --exp exp... --eval_alg srp_phat
```

Or want to get the result of one sample
```
python eval_runner_main.py --exp exp... --example --sample_name sample_0 --eval_alg srp_phat
```

The results are saved in **Data/exp../srp_phat_eval.json** or **Data/exp../eval_example** for example.
