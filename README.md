# nlg-sclstm-multiwoz
pytorch implementation of semantically-conditioned LSTM on multiwoz data


semantically-conditioned LSTM: https://arxiv.org/pdf/1508.01745.pdf

# Run the code

dataSplitFile=resource/woz3/data_split/Boo_ResDataSplitRand0925.json

l=1

bs=200

lr=0.001

model_path=./sclstm.pt

log=./sclstm.log

res=./sclstm.res



TRAIN

python3 run_woz3.py --data_split=$dataSplitFile --mode=train --model_path=$model_path --n_layer=$l --bs=$bs --lr=$lr > $log



TEST

python3 run_woz3.py --data_split=$dataSplitFile --mode=test --model_path=$model_path --n_layer=$l --beam_size=10 --bs=$bs > $res
