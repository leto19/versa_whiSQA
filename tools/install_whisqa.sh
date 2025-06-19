
rm -rf WhiSQA

mkdir -p versa/utterance_metrics/WhiSQA
mkdir -p versa_cache/WhiSQA
git clone https://github.com/leto19/WhiSQA


mv WhiSQA/checkpoints/ versa_cache/WhiSQA/
mv WhiSQA/models/mel_filters.npz versa_cache/WhiSQA/

rm -rf WhiSQA