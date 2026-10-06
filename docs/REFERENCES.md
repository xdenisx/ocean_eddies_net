# Implementation references

Primary documentation consulted for API/data contracts (not evidence of improved
performance on this dataset):

- PyTorch CrossEntropyLoss — class-index targets, ignore_index and class weighting:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch serialization — loading tensor/dictionary checkpoints:
  https://docs.pytorch.org/docs/stable/notes/serialization.html
- Segmentation Models PyTorch — classes/in_channels and model constructors:
  https://smp.readthedocs.io/en/latest/models.html
- Hugging Face UPerNet — constructor versus from_pretrained, Swin backbone and outputs:
  https://huggingface.co/docs/transformers/model_doc/upernet
- Hugging Face SegFormer:
  https://huggingface.co/docs/transformers/model_doc/segformer
- Rasterio masks — nodata metadata, valid data masks and internal/external masks:
  https://rasterio.readthedocs.io/en/stable/topics/masks.html
- scikit-image exposure — equalize_hist and equalize_adapthist:
  https://scikit-image.org/docs/stable/api/skimage.exposure.html

The supplied legacy files, not outside documentation, are the source of the
custom TransUNet-like/HRNet-fusion implementations and historical CLI names.
Their exact copies and hashes are included in legacy/.
