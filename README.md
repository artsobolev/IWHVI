# Importance Weighted Hierarchical Variational Inference

This repo contains source code for the [Importance Weighted Hierarchical Variational Inference](https://arxiv.org/abs/1905.03290) paper (NeurIPS 2019).

To train a 32-D MNIST VAE use the following command:
```
python ./iwhvae_run.py                                              \
    --annealing_factor                   0.95                       \
    --annealing_speed                    100                        \
    --dataset                            dynamic_mnist              \
    --z_dim                              32                         \
    --noise_dim                          32                         \
    --decoder_arch                       300 300                    \
    --encoder_arch                       300 300                    \
    --tau_arch                           300 300                    \
    --tau_use_dreg                                                  \
    --train_batch_size                   256                        \
    --learning_rate                      0.001                      \
    --epochs                             10000                      \
    --evaluate_every                     25                         \
    --q_kl_warm_up_cycles                1                          \
    --q_kl_warm_up_fraction              0.03                       \
    --save_every                         100000                     \
    --train_iwhvi_samples_step_schedule  250,5 500,25 1000,50       \
    --val_batch_size                     50000                      \
    --val_iwae_samples                   1000                       \
    --val_iwhvi_samples                  100                        \
    --save_path                          ./exp_dynamic_mnist_d32_lr1e-3_klwu300_tklwu1_tdl1
```

To evaluate a trained model, use the following command:
```
python ./iwhvae_eval.py                                             \
    --n_repeats                          10                         \
    --dataset                            dynamic_mnist              \
    --z_dim                              32                         \
    --noise_dim                          32                         \
    --decoder_arch                       300 300                    \
    --encoder_arch                       300 300                    \
    --tau_arch                           300 300                    \
    --train_batch_size                   256                        \
    ./exp_dynamic_mnist_d32_lr1e-3_klwu300_tklwu1_tdl1/model-weights-final
```

If you're getting "out of memory" exceptions, add `--test_iwae_batch_size N` with `N` smaller than 5000.

## Related Links

[ [Preprint](https://arxiv.org/abs/1905.03290) | [Blogpost](http://artem.sobolev.name/posts/2019-05-10-importance-weighted-hierarchical-variational-inference.html) | [Talk](https://youtu.be/pdSu7XfGhHw) | [Toy Example in Colab](https://colab.research.google.com/drive/1slWtEve2M4ogvoa3TLD_TFI4OBCVXJj3) ]
