GLOBALLEN="4096" # the maximum sequence length of the model
MAXCTXLEN="4064" # the maximum input context length
GENLEN="32" # the maximun generation length

SEED=42
DEVICE="0,1" # the GPU device id
TOPP="0.0" # top-p sampling, set to 0.0 for greedy decoding
GPUS=2 # number of gpus
FLAG="no" # set to "yes" to enable int4 quantization to load the model

TESTFILE="fin|$1"
bash run_group_decode_fileio.sh $SEED $DEVICE $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP $GPUS $FLAG