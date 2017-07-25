#!/bin/bash

rm -f data/test_data.csv
rm -f data/train_data.csv
rm -f data/valid_data.csv
rm -f data/*.npz

echo 'Prepping data...'
python prep_data.py

echo 'Preprocessing data...'
python fit_tsne.py

models=('fm' 'gbt' 'lr' 'pairwise')

echo 'Running models...'
for m in ${models[@]}; do
    python models/pipeline/$m.py
done

echo 'Ensembling all predictions...'
python ensemble.py

echo 'Automation finished'
