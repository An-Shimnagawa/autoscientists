#!/bin/bash
set -e

mkdir -p kermut/data
cd kermut

# Precomputed resources: ESM-2 embeddings, ProteinMPNN conditional probs,
# Cα coordinates, and zero-shot fitness predictions (~3.9 GB)
curl -o kermut_data.zip https://sid.erda.dk/share_redirect/c2EWrbGSCV/kermut_data.zip
unzip kermut_data.zip && rm kermut_data.zip

# DMS assay data with CV fold assignments (cv_folds_singles_substitutions/)
# Source: ProteinGym v1.3 — https://github.com/petergroth/kermut
curl --insecure -o cv_folds_singles_substitutions.zip \
  https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip
unzip cv_folds_singles_substitutions.zip -d data
rm cv_folds_singles_substitutions.zip