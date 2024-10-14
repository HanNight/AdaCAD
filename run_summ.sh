GLOBALLEN="4096"
MAXCTXLEN="3996"
GENLEN="100"

SEED=42
DEVICE="0,1" # the GPU device id
TOPP="0.0" # top-p sampling, set to 0.0 for greedy decoding
GPUS=2 # number of gpus
FLAG="no" # set to "yes" to enable int4 quantization to load the model
THRESHOLD="0.3" # warmup operation for long-form generation

TESTFILE="fin|$1"
bash run_group_decode_fileio.sh $SEED $DEVICE $TESTFILE $GLOBALLEN $MAXCTXLEN $GENLEN $TOPP $GPUS $FLAG $THRESHOLD