#!/bin/bash
if (($# != 2))
then
    echo "The total number of command line arguments should be 2, in the form: ./run.sh test_file prediction_file"
else
    iconv -f ISO-8859-1 -t UTF-8 train.txt > train_utf8.txt
    iconv -f ISO-8859-1 -t UTF-8 validation.txt > validation_utf8.txt
    iconv -f ISO-8859-1 -t UTF-8 $1 > test_utf8.txt
    python3 ./spanish_ner.py test_utf8.txt $2
fi