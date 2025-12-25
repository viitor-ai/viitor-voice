mkdir -p checkpoints
# Users in mainland China, please uncomment the line below.
# export HF_ENDPOINT=https://hf-mirror.com

# codec
hf download facebook/w2v-bert-2.0 --local-dir checkpoints/w2v
hf download amphion/dualcodec --local-dir checkpoints/dualcodec

# viitor_voice
hf download ZzWater/ViiTor-voice-2.0-base --local-dir checkpoints/viitor