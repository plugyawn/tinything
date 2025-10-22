## `tinythings`: tiny implementations in `tinygrad` 

Currently a small(er) nanoGPT, written in tinygrad, from scratch. 
Tinygrad allows acceleration on virtually any backend: probably the simplest way to write accelerated training on modern Macs.

- [x] CausalAttention with RoPE [layers/attention.py]
- [x] Basic FFNs [layers/feedforward.py]
- [x] SwiGLU FFNs [layers/feedforward.py]
- [ ] Mixture of Experts [layers/moe.py]
- [x] SGD [optimizers/sgd.py]
- [ ] Adam [optimizers/sgd.py]
- [ ] Muon [optimizers/muon.py]
- [x] LayerNorm [utils/transformer_methods.py]
- [x] Cross Entropy [utils/loss_functions.py]
- [x] Naive character-level tokenization [utils/dataloader.py]
- [ ] Byte-pair encoding [utils/dataloader.py]
- [ ] Diffusion text modelling 

