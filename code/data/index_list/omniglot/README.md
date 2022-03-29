### How to use the index files for the experiments ?

The index files are named like `support_batch_T.txt`, where x indicates the session number. Each index file stores the indexes of the images that are selected for the session.
"support_batch_1.txt" stores all the base class training images (1200*5=6000). Each `support_batch_T.txt` (T>1) stores the 235 (47 classes and 5 shots per class) few-shot new class training images.
You may adopt the following steps to perform the experiments.

First, at session 1, train a base model using the images in `support_batch_1.txt`;

Then, at session T (T>1), finetune the model trained at the previous session (T-1), only using the images in support_batch_T.txt.

For evaluating the model at session T, use the files indexed in `query_batch_T.txt`.
