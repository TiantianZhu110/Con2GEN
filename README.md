# Con2GEN
--
&ensp;&ensp;&ensp;&ensp;code for EMNLP 2023 paper: **Controllable Contrastive Generation for Multilingual Biomedical Entity Linking**

#Requirement
--
```
python>=3.7
pytorch>=1.6
fairseq>=0.10
transformers>=4.2 (optional for inference of GENRE)
```
you can go [here](https://github.com/nicola-decao/fairseq/tree/fixing_prefix_allowed_tokens_fn) to know more about fairseq. 


# How to run the code?
-- 

1.download train data (We will upload the data later), and put it to the folder "data/origin\_data and data/cl\_data".

2.Run the file "train\_lang\_type.sh" for training to obtain the model into folder "model/finetune\_lang\_type".

3.Run the file "train\_cl.sh for contrastive learning and fine-tuning"

4.Run gener\_predict\_typelist.py to predict the result.

5.Run acc.py to calculated result indicators.
